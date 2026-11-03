//! `fp8_block128`: a release's FP8 projections as shipped. A projection is e4m3fn codes, U8 `[out, in]` `.weight`, with
//! one f32 scale per 64 output rows x 128 input columns, F32 `[ceil(out / 64), ceil(in / 128)]` `.weight_scale_inv`;
//! value = code x scale (MiMo-V2.6-Flash's fp8res pack, `{"mode":"fp8_e4m3","bits":8,"block":[64,128]}` per site). The
//! read (`linear`, the shared Linear's arm for a U8 weight beside its F32 scale grid): up to 8 bf16 rows, one GEMV
//! (`kernels/linear/fp8_block128_gemv.metal`, f32 code x scale, no weight rounding; its plan in `fp8_block128_gemv.zig`);
//! wider, the weight as bf16(code x scale) (one f32 product, one BF16 rounding) under MLX's matmul. Packs declare it per
//! projection inside an MLX `quantization` block, so `native` loads it; this table serves a block that names it alone.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const native = @import("native.zig");
const plan = @import("fp8_block128_gemv.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;

const Str = abi.Str;
const Verdict = abi.Verdict;

pub const block_rows: u32 = plan.scale_rows;
pub const block_cols: u32 = plan.scale_cols;

const header = kernelSource("kernels/linear/fp8_block128_header.metal");
const gemv_source = kernelSource("kernels/linear/fp8_block128_gemv.metal");
const dequant_source = kernelSource("kernels/linear/fp8_block128_dequant.metal");

/// U8 e4m3 codes `[out, in]` beside an F32 scale grid `[ceil(out / 64), ceil(in / 128)]` (Sushi's row-scaled e4m3 has
/// one scale row per code row).
pub fn isBlockScaled(w: mlx.mlx_array, sc: mlx.mlx_array) bool {
    if (w.ctx == null or sc.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint8 or mlx.mlx_array_dtype(sc) != .float32) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    return ws.len == 2 and ss.len == 2 and ws[0] > 1 and
        ss[0] == @divFloor(ws[0] + block_rows - 1, block_rows) and ss[1] == @divFloor(ws[1] + block_cols - 1, block_cols);
}

/// The kernels read the scale grid as `device` memory; MLX binds an input under 8 elements as `constant`.
fn kernelsTake(sc: mlx.mlx_array, s: mlx.mlx_stream) bool {
    return mlx.mlx_array_size(sc) >= 8 and mlx.streamIsGpu(s);
}

/// The kernels, made once on first use (inference thread).
var gemv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var dequant_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, out: [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{out};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.Fp8Block128Kernel;
    slot.* = k;
    return k;
}

fn applyKernel(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const in_vec = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, in_vec, c, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

/// `x @ W^T` for codes `w` `[n, k]` and their scale grid `sc`: the GEMV at up to 8 bf16 rows where its plan covers the
/// shape, else the prompt route (`matmulInto`). `x` is `[..., k]`.
pub fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| m *= d;
    const site = plan.plan(@intCast(ws[0]), @intCast(ws[1]));
    if (m > plan.max_rows or site == null or mlx.mlx_array_dtype(x) != .bfloat16 or !kernelsTake(sc, s)) return matmulInto(w, sc, x, s);
    const p = plan.planAt(site.?, @intCast(ws[0]), @intCast(ws[1]), @intCast(m));
    const l = plan.launch(p, ws[0]);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ m, ws[0] }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "M", m }, .{ "N", ws[0] }, .{ "K", ws[1] }, .{ "R", p.r }, .{ "KS", p.ks }, .{ "RG", p.rg } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ m, ws[1] }, 2, s));
    const y = try applyKernel(try kernel(&gemv_kernel, "fp8_block128_gemv", &.{ "x", "w", "scales" }, "y", gemv_source), c, &.{ x2, w, sc }, s);
    if (xs.len == 2) return y;
    defer _ = mlx.mlx_array_free(y);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    shape[xs.len - 1] = ws[0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, xs.len, s));
    return out;
}

/// The block names this format: e4m3 codes, 8 bits, the 64 x 128 scale grid.
fn ownBlock(block_json: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), block_json, .{}) catch return false;
    return v == .object and plan.isEntry(v.object);
}

fn claims(block_json: Str, _: Str) callconv(.c) abi.Claim {
    return if (ownBlock(block_json.slice())) .by_type else .no;
}

fn validate(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    return if (ownBlock(block_json.slice())) .ok else Verdict.refuse(.unsupported, "fp8_block128: a block other than e4m3, 8 bits, [64, 128]");
}

fn open(_: *abi.LoadCtx) callconv(.c) Verdict {
    return .ok;
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

/// `<name>.weight` U8 `[out, in]` and `<name>.weight_scale_inv` F32 on the 64 x 128 grid, shared with the source.
fn loadInto(ctx: *const abi.LoadCtx, name: []const u8, out: *abi.Weight) !void {
    const w = native.tensor(ctx, name, "weight");
    const sc = native.tensor(ctx, name, "weight_scale_inv");
    if (w.ctx == null or sc.ctx == null) return error.MissingWeight;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (ws.len != 2 or ss.len != 2 or mlx.mlx_array_dtype(w) != .uint8 or mlx.mlx_array_dtype(sc) != .float32) return error.Fp8Block128Layout;
    const rows: u32 = @intCast(ws[0]);
    const cols: u32 = @intCast(ws[1]);
    if (ss[0] != (rows + block_rows - 1) / block_rows or ss[1] != (cols + block_cols - 1) / block_cols) return error.Fp8Block128ScaleGrid;
    out.* = .{ .bits = 8, .group_size = block_cols, .block = block_rows };
    errdefer native.release(out);
    var hw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&hw, w));
    out.w = native.wrap(hw);
    var hs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&hs, sc));
    out.scales = native.wrap(hs);
}

fn loadTensor(ctx: *const abi.LoadCtx, name: Str, _: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadInto(ctx, name.slice(), out) catch |e| return native.failed(e);
    return .ok;
}

fn loadBank(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "fp8_block128: expert banks");
}

/// The weight as bf16(code x scale), `[out, in]`: e4m3 exactly in f32, times its tile's scale (one f32 rounding), one
/// BF16 rounding. One kernel on Metal over whole column blocks; MLX ops elsewhere (the same values).
pub fn dequantize(codes: mlx.mlx_array, scales: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(codes);
    if (kernelsTake(scales, s) and @rem(sh[1], block_cols) == 0) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(sh[0] * sh[1], 4), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "N", sh[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", sh[1]));
        return applyKernel(try kernel(&dequant_kernel, "fp8_block128_dequant", &.{ "w", "scales" }, "wb", dequant_source), c, &.{ codes, scales }, s);
    }
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_from_fp8(&f, codes, .float32, s));
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_repeat_axis(&r, scales, block_rows, 0, s));
    var rc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rc);
    try mlx.check(mlx.mlx_repeat_axis(&rc, r, block_cols, 1, s));
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_slice(&grid, rc, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ sh[0], sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_multiply(&p, f, grid, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, p, .bfloat16, s));
    return out;
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const y = linear(native.arr(x), native.arr(w.w), native.arr(w.scales), .{ .ctx = s.ctx }) catch |e| return native.failed(e);
    out.* = native.wrap(y);
    return .ok;
}

/// The prompt route: `x @ bf16(code x scale)^T` under MLX's matmul.
fn matmulInto(codes: mlx.mlx_array, scales: mlx.mlx_array, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const wb = try dequantize(codes, scales, s);
    defer _ = mlx.mlx_array_free(wb);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wb, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, wt, s));
    return y;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "fp8_block128: expert banks");
}

pub const table: abi.QuantTable = .{
    .name = Str.of("fp8_block128"),
    .claims = &claims,
    .validate = &validate,
    .bill = &native.billStored,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &native.release,
    .matmul = &matmul,
    .moe = &moe,
};

// ── tests ──

const testing = std.testing;

test "fp8_block128: claims and validates its own block only" {
    const own = "{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}";
    try testing.expectEqual(abi.Claim.by_type, claims(.of(own), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}"), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}"), .of("{}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{}"), .of("{}")));
}

fn codesOf(r: std.Random, out: []u8) void {
    for (out) |*c| c.* = while (true) {
        const v = r.int(u8);
        if (v & 0x7f != 0x7f) break v;
    };
}

fn bf16Of(r: std.Random, out: []u16) void {
    for (out) |*b| b.* = plan.f32ToBf16(1.5 * r.floatNorm(f32));
}

fn evalBf16(y: mlx.mlx_array) ![]const u16 {
    try mlx.check(mlx.mlx_array_eval(y));
    return (mlx.mlx_array_data_bfloat16(y) orelse return error.NoData)[0..mlx.mlx_array_size(y)];
}

// Bar: bf16(code x scale) of the host, on the kernel and on MLX's ops, ragged tiles on both axes included.
test "fp8_block128: the prompt route's weight is bf16(code x scale), each code on its own block's scale" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const n = 140;
    var rnd = std.Random.DefaultPrng.init(64);
    // 200 columns: ragged column blocks (MLX's ops); 384: whole blocks (the kernel on Metal: 9 scales).
    inline for (.{ 200, 384 }) |k| {
        const cols = (k + block_cols - 1) / block_cols;
        var codes: [n * k]u8 = undefined;
        codesOf(rnd.random(), &codes);
        var scales: [3 * cols]f32 = undefined;
        for (&scales) |*v| v.* = std.math.ldexp(1.0 + rnd.random().float(f32), -6 - rnd.random().intRangeAtMost(i32, 0, 3));
        const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ n, k }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ 3, cols }, 2, .float32);
        defer _ = mlx.mlx_array_free(sc);
        for ([_]mlx.mlx_stream{ mlx.gpuStream(), mlx.mlx_default_cpu_stream_new() }) |s| {
            defer _ = mlx.mlx_stream_free(s);
            const got = try dequantize(w, sc, s);
            defer _ = mlx.mlx_array_free(got);
            const g = try evalBf16(got);
            for (0..n) |r| for (0..k) |c| {
                try testing.expectEqual(plan.f32ToBf16(plan.weight32(codes[r * k + c], scales[(r / block_rows) * cols + c / block_cols])), g[r * k + c]);
            };
        }
    }
}

// Bar: every output is the kernel-order host sum (`fp8_block128_gemv.hostDot`) rounded once, at 1..8 rows and K split
// 1, 4 and 8 ways; 9 rows take the prompt route (MLX's matmul over the bf16 weight).
test "fp8_block128: the GEMV is the kernel-order host sum at every width up to 8 rows; wider rows take the prompt route" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var rnd = std.Random.DefaultPrng.init(0xf8b1);
    const r = rnd.random();
    for ([_][2]u32{ .{ 4096, 512 }, .{ 1024, 1024 }, .{ 256, 4096 } }, [_]c_int{ 1, 4, 8 }) |nk, ks| {
        const n = nk[0];
        const k = nk[1];
        const site = plan.plan(n, k).?;
        try testing.expectEqual(ks, site.ks);
        const codes = try a.alloc(u8, n * k);
        defer a.free(codes);
        codesOf(r, codes);
        const scales = try a.alloc(f32, (n / block_rows) * (k / block_cols));
        defer a.free(scales);
        for (scales) |*v| v.* = std.math.ldexp(1.0 + r.float(f32), -17 + @as(i32, @intCast(r.uintLessThan(u32, 11))));
        const xs = try a.alloc(u16, 9 * k);
        defer a.free(xs);
        bf16Of(r, xs);
        const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        const sc = mlx.mlx_array_new_data(scales.ptr, &[_]c_int{ @intCast(n / block_rows), @intCast(k / block_cols) }, 2, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const wrow = try a.alloc(f32, k);
        defer a.free(wrow);
        // The host sums of every output of the 8 rows (the GEMV's rows are M-invariant).
        const want = try a.alloc(u16, 8 * n);
        defer a.free(want);
        for (0..n) |j| {
            for (wrow, 0..) |*v, i| v.* = plan.weight32(codes[j * k + i], scales[(j / block_rows) * (k / block_cols) + i / block_cols]);
            for (0..8) |m| want[m * n + j] = plan.f32ToBf16(plan.hostDot(site, xs[m * k ..][0..k], wrow));
        }
        for (1..10) |m| {
            const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ 1, @intCast(m), @intCast(k) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const y = try linear(x, w, sc, s);
            defer _ = mlx.mlx_array_free(y);
            try testing.expectEqualSlices(c_int, &.{ 1, @intCast(m), @intCast(n) }, mlx.getShape(y));
            const got = try evalBf16(y);
            if (m <= plan.max_rows) {
                try testing.expectEqualSlices(u16, want[0 .. m * n], got);
                continue;
            }
            const x2 = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(m), @intCast(k) }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x2);
            const p = try matmulInto(w, sc, x2, s);
            defer _ = mlx.mlx_array_free(p);
            try testing.expectEqualSlices(u16, try evalBf16(p), got);
        }
    }
}
