//! MoE prefill over fp expert banks (nvfp4, mxfp8) on MLX's CUDA backend: the sorted gather GEMM,
//! gate+up+SwiGLU in one launch, and the weighted top-k sum. Weights stream through registers four
//! 128-wide K stages ahead, each warp load reading 64+ contiguous bytes of a row (32-byte row pieces,
//! as `qmm_sm80` reads them, run at half the DRAM bandwidth); the CTA turns each stage into a bf16
//! tile once, and `ldmatrix` feeds it to the same `mma.m16n8k16` in the same K order, so every output
//! is bit-identical to the chain's (the weights are exact in bf16). Where MLX would not run `qmm_sm80`
//! the kernels decline.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const Format = enum(c_int) { mxfp8 = 1, nvfp4 = 3 };

const BM = 64;
const BN = 128;
const BK = 128;
const STAGES = 2;
/// 128-wide K stages of weights held in registers ahead of their use.
const PREFETCH = 4;
const THREADS = 256;

fn sharedBytes() c_int {
    return STAGES * BM * BK * 2 + BN * BK * 2 + 1024;
}

const HEADER = kernelSource("kernels/moe/moe_prefill_cuda_header.cu");

// Grid (THREADS * N/128, E, splits): CTA (nt, e, z) computes y columns [nt*128, +128) for expert e's
// rows in 64-row tiles z, z+splits, ...
const SOURCE = kernelSource("kernels/moe/moe_prefill_cuda.cu");

// Grid (THREADS * I/64, E, splits): CTA (nt, e, z) computes act columns [nt*64, +64) of expert e's rows,
// B rows 0-63 from the gate bank and 64-127 from the up bank; x rows come through the routing map `rows`.
// The epilogue is the chain's: each projection rounded to T, gate times its expert scale, then
// silu(g) * u with MLX's own functors in that order.
const GU_SOURCE = kernelSource("kernels/moe/moe_prefill_cuda_gu.cu");

/// MLX's sorted gather runs `qmm_sm80` only from 16 rows and 4 rows per expert on average
/// (`supports_gather_qmm_rhs_sm80`); below that its mat-vec sums in another order, so we decline too.
fn mlxTakesExpertPath(rows: c_int, experts: c_int) bool {
    return rows >= 16 and experts > 0 and @divTrunc(rows, experts) >= 4;
}

var kernels: [2]?mlx.mlx_fast_cuda_kernel = .{ null, null };
var engaged = false;
/// Gathers served (the tests assert the path engaged).
pub var served: u64 = 0;

fn kernelFor(fmt: Format) !mlx.mlx_fast_cuda_kernel {
    const slot = &kernels[if (fmt == .mxfp8) 0 else 1];
    if (slot.*) |k| return k;
    const ins = [_][*:0]const u8{ "x", "w", "s", "inds" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{"y"};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const name: [*:0]const u8 = if (fmt == .mxfp8) "msv_moe_prefill_mxfp8" else "msv_moe_prefill_nvfp4";
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, SOURCE, HEADER, true, sharedBytes());
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    slot.* = k;
    return k;
}

/// `gather_qmm(x, w, s, rhs_indices=inds, sorted)` for bf16 rows: x [R, 1, K] or [R, K], banks
/// w [E, N, K*bits/32] uint32 and s [E, N, K/group] uint8, inds [R] uint32 sorted. Returns
/// [R, 1, N] (x's rank), or null outside the kernel's envelope.
pub fn sortedGather(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, inds: mlx.mlx_array, fmt: Format) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(inds) != .uint32) return null;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .uint8) return null;
    const bits: c_int = if (fmt == .mxfp8) 8 else 4;
    const group: c_int = if (fmt == .nvfp4) 16 else 32;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const is = mlx.getShape(inds);
    if (is.len != 1 or ws.len != 3 or (xs.len != 2 and !(xs.len == 3 and xs[1] == 1))) return null;
    const R = is[0];
    const K = xs[xs.len - 1];
    const E = ws[0];
    const N = ws[1];
    if (xs[0] != R or ws[2] * 32 != K * bits or !mlxTakesExpertPath(R, E)) return null;
    if (@rem(K, BK) != 0 or @rem(N, BN) != 0) return null;
    if (mlx.getShape(sc)[2] * group != K) return null;

    const n_tiles = @divExact(N, BN);
    const tiles_m = @divTrunc(R + BM - 1, BM);
    const splits = std.math.clamp(@divTrunc(16384, n_tiles * E), 1, 8);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ R, N }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, THREADS * n_tiles, E, @min(splits, tiles_m)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, THREADS, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "FMT", @intFromEnum(fmt)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "K", K));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "N", N));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "STG", STAGES));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "BDEP", PREFETCH));
    const ins = [_]mlx.mlx_array{ x, w, sc, inds };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, try kernelFor(fmt), v, c, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (xs.len == 3) try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ R, 1, N }, 3, s)) else try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ R, N }, 2, s));
    if (!engaged) {
        engaged = true;
        log.info("[moe] CUDA prefill expert GEMM engaged: {s} K={d} N={d} experts={d}\n", .{ @tagName(fmt), K, N, E });
    }
    served += 1;
    return out;
}

var gu_kernels: [2]?mlx.mlx_fast_cuda_kernel = .{ null, null };
var gu_engaged = false;
/// Fused gate+up launches served (the tests assert the path engaged).
pub var gu_served: u64 = 0;

/// The sorted prefill's `silu(gate(x) * gscale) * up(x)` in one launch: x [T, K] token rows read
/// through `rows` [R] (each routed row's token), `inds` [R] sorted experts, gate/up banks
/// [E, I, K*bits/32] uint32 with uint8 scales, `gscale` [E] or null. Returns act [R, I], or null
/// outside the kernel's envelope.
pub fn gateUp(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, inds: mlx.mlx_array, gw: mlx.mlx_array, gsc: mlx.mlx_array, uw: mlx.mlx_array, usc: mlx.mlx_array, gscale: ?mlx.mlx_array, fmt: Format) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(rows) != .uint32) return null;
    if (gscale) |g| if (mlx.mlx_array_dtype(g) != .bfloat16) return null;
    inline for (.{ gw, uw }) |b| if (mlx.mlx_array_dtype(b) != .uint32) return null;
    inline for (.{ gsc, usc }) |b| if (mlx.mlx_array_dtype(b) != .uint8) return null;
    const bits: c_int = if (fmt == .mxfp8) 8 else 4;
    const group: c_int = if (fmt == .nvfp4) 16 else 32;
    const xs = mlx.getShape(x);
    const gsh = mlx.getShape(gw);
    const is = mlx.getShape(inds);
    if (xs.len != 2 or gsh.len != 3 or is.len != 1 or mlx.getShape(rows).len != 1 or mlx.getShape(rows)[0] != is[0]) return null;
    if (!std.mem.eql(c_int, gsh, mlx.getShape(uw))) return null;
    const R = is[0];
    const K = xs[1];
    const E = gsh[0];
    const I = gsh[1];
    if (gsh[2] * 32 != K * bits or @rem(K, BK) != 0 or @rem(I, 64) != 0 or !mlxTakesExpertPath(R, E)) return null;
    if (mlx.getShape(gsc)[2] * group != K or mlx.getShape(usc)[2] * group != K) return null;

    const slot = &gu_kernels[if (fmt == .mxfp8) 0 else 1];
    if (slot.* == null) {
        const ins = [_][*:0]const u8{ "x", "rows", "inds", "gw", "gs", "uw", "us", "gscale" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const outs = [_][*:0]const u8{"act"};
        const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const name: [*:0]const u8 = if (fmt == .mxfp8) "msv_moe_prefill_gate_up_mxfp8" else "msv_moe_prefill_gate_up_nvfp4";
        const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, GU_SOURCE, HEADER, true, sharedBytes());
        if (k.ctx == null) return error.CudaKernelCreateFailed;
        slot.* = k;
    }
    const n_tiles = @divExact(I, 64);
    const tiles_m = @divTrunc(R + BM - 1, BM);
    const splits = std.math.clamp(@divTrunc(16384, n_tiles * E), 1, 8);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ R, I }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, THREADS * n_tiles, E, @min(splits, tiles_m)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, THREADS, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    inline for (.{ .{ "FMT", @intFromEnum(fmt) }, .{ "K", K }, .{ "I", I }, .{ "STG", STAGES }, .{ "BDEP", PREFETCH }, .{ "GSCALE", @as(c_int, @intFromBool(gscale != null)) } }) |a|
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const ins = [_]mlx.mlx_array{ x, rows, inds, gw, gsc, uw, usc, gscale orelse x };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, slot.*.?, v, c, s));
    var act = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_vector_array_get(&act, o, 0));
    if (!gu_engaged) {
        gu_engaged = true;
        log.info("[moe] CUDA prefill gate+up+SwiGLU engaged: {s} K={d} I={d} experts={d}\n", .{ @tagName(fmt), K, I, E });
    }
    gu_served += 1;
    return act;
}

// Block per token, thread per 8 output columns: the chain's unsort, routing-weight Multiply and
// sequential bf16 Sum over the top-k (MLX's col_reduce_small), k in order.
const REDUCE_SOURCE = kernelSource("kernels/moe/moe_prefill_cuda_reduce.cu");

var reduce_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var reduce_engaged = false;

/// sum_k scores[b,s,k] * down[inverse[(b*S+s)*K + k]]: the routed rows back in token order,
/// weighted and summed over the top-k, [B, S, H]. down [B*S*K, H] and scores [B, S, K] bf16,
/// inverse [B*S*K] uint32. Null outside the kernel's envelope.
pub fn weightedSum(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const is = mlx.getShape(inverse);
    if (ds.len != 2 or ss.len != 3 or is.len != 1) return null;
    const tokens = ss[0] * ss[1];
    const K = ss[2];
    const H = ds[1];
    if (tokens < 1 or ds[0] != tokens * K or is[0] != ds[0] or @rem(H, 8) != 0 or @divExact(H, 8) > 1024) return null;
    if (reduce_kernel == null) {
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const outs = [_][*:0]const u8{"y"};
        const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_cuda_kernel_new("msv_moe_prefill_sum", in_vec, out_vec, REDUCE_SOURCE, HEADER, true, 0);
        if (k.ctx == null) return error.CudaKernelCreateFailed;
        reduce_kernel = k;
    }
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ ss[0], ss[1], H }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, tokens * @divExact(H, 8), 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, @divExact(H, 8), 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "TOPK", K));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "H", H));
    const ins = [_]mlx.mlx_array{ down, inverse, scores };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, reduce_kernel.?, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    if (!reduce_engaged) {
        reduce_engaged = true;
        log.info("[moe] CUDA prefill weighted sum engaged: top-{d} H={d}\n", .{ K, H });
    }
    return y;
}

const testing = std.testing;

/// Sorted expert ids for `rows` rows over `experts`: uneven, with empty experts and ones past a tile.
fn sortedIds(a: std.mem.Allocator, rows: usize, experts: u32, seed: u64) ![]u32 {
    const ids = try a.alloc(u32, rows);
    var prng = std.Random.DefaultPrng.init(seed);
    for (ids) |*v| {
        const r = prng.random();
        // A quarter of the rows crowd into four experts, so some take several BM-row tiles.
        v.* = if (r.intRangeLessThan(u32, 0, 4) == 0) r.intRangeLessThan(u32, 0, 4) * (experts / 4) else r.intRangeLessThan(u32, 0, experts / 2) * 2;
    }
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
    return ids;
}

test "moe_prefill_cuda: the sorted expert GEMM is bit-identical to gather_qmm (nvfp4, mxfp8)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cases = [_]struct { fmt: Format, mode: [*:0]const u8, group: c_int, bits: c_int, k: c_int, n: c_int, rows: usize }{
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4, .k = 2560, .n = 640, .rows = 1300 },
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4, .k = 640, .n = 2560, .rows = 700 },
        .{ .fmt = .mxfp8, .mode = "mxfp8", .group = 32, .bits = 8, .k = 2560, .n = 640, .rows = 1300 },
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4, .k = 256, .n = 128, .rows = 128 },
    };
    const E: c_int = 32;
    for (cases, 0..) |cs, ci| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 11 + ci));
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ E, cs.n, cs.k }, 3, .bfloat16, 0, 0.05, key, s));
        var q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(q);
        try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_vector_array_get(&w, q, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
        const rows: c_int = @intCast(cs.rows);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ rows, 1, cs.k }, 3, .bfloat16, 0, 1, key, s));
        const ids = try sortedIds(testing.allocator, cs.rows, E, ci);
        defer testing.allocator.free(ids);
        const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        const before = served;
        const got = (try sortedGather(s, x, w, sc, inds, cs.fmt)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqual(before + 1, served);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_gather_qmm(&ref, x, w, sc, .{}, .{}, inds, true, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, true, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        if (!same) std.debug.print("moe_prefill_cuda {s} K={d} N={d} rows={d}: differs from gather_qmm\n", .{ cs.mode, cs.k, cs.n, cs.rows });
        try testing.expect(same);
    }
}

test "moe_prefill_cuda: below 4 rows per expert MLX's sorted mat-vec is bit-identical to its per-row one (nvfp4, mxfp8)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const E: c_int = 32;
    const K: c_int = 1280;
    const N: c_int = 640;
    const R: c_int = 100;
    const cases = [_]struct { fmt: Format, mode: [*:0]const u8, group: c_int, bits: c_int }{
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4 },
        .{ .fmt = .mxfp8, .mode = "mxfp8", .group = 32, .bits = 8 },
    };
    for (cases, 0..) |cs, ci| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 41 + ci));
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ E, N, K }, 3, .bfloat16, 0, 0.05, key, s));
        var q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(q);
        try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_vector_array_get(&w, q, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ R, 1, K }, 3, .bfloat16, 0, 1, key, s));
        const ids = try sortedIds(testing.allocator, @intCast(R), E, ci + 9);
        defer testing.allocator.free(ids);
        const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{R}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        try testing.expect((try sortedGather(s, x, w, sc, inds, cs.fmt)) == null);
        var outs: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        for (&outs, [_]bool{ true, false }) |*o, sorted|
            try mlx.check(mlx.mlx_gather_qmm(o, x, w, sc, .{}, .{}, inds, true, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, sorted, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, outs[0], outs[1], false, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    }
}

test "moe_prefill_cuda: below 4 rows per expert MLX's gather takes its mat-vec, so the kernels decline" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const E: c_int = 32;
    const K: c_int = 256;
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ E, 128, 32 }, 3, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_zeros(&sc, &[_]c_int{ E, 128, 16 }, 3, .uint8, s));
    for ([_]c_int{ 127, 15 }) |rows| {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ rows, K }, 2, .bfloat16, s));
        const ids = try testing.allocator.alloc(u32, @intCast(rows));
        defer testing.allocator.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i * 32 / @as(usize, @intCast(rows)));
        const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        try testing.expect((try sortedGather(s, x, w, sc, inds, .nvfp4)) == null);
        try testing.expect((try gateUp(s, x, inds, inds, w, sc, w, sc, null, .nvfp4)) == null);
    }
}

test "moe_prefill_cuda: the weighted top-k sum is bit-identical to take + multiply + sum" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const T_: c_int = 37;
    const K: c_int = 10;
    const H: c_int = 2560;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 21));
    var down = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down);
    try mlx.check(mlx.mlx_random_normal(&down, &[_]c_int{ T_ * K, H }, 2, .bfloat16, 0, 1, key, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_random_uniform(&scores, mlx.mlx_array_new_float(0), mlx.mlx_array_new_float(0.4), &[_]c_int{ 1, T_, K }, 3, .bfloat16, key, s));
    // A permutation of the routed rows, as the counting sort's inverse order is.
    var perm_data: [370]u32 = undefined;
    for (&perm_data, 0..) |*v, i| v.* = @intCast((i * 151 + 7) % 370);
    const inv = mlx.mlx_array_new_data(&perm_data, &[_]c_int{T_ * K}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inv);
    const got = (try weightedSum(s, down, inv, scores)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(got);
    var unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unsorted);
    try mlx.check(mlx.mlx_take_axis(&unsorted, down, inv, 0, s));
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, unsorted, &[_]c_int{ 1, T_, K, H }, 4, s));
    var se = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(se);
    try mlx.check(mlx.mlx_expand_dims(&se, scores, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, r4, se, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_sum_axis(&ref, weighted, -2, false, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

test "moe_prefill_cuda: fused gate+up+SwiGLU is bit-identical to the chain (nvfp4, mxfp8; with and without a gate scale)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const E: c_int = 32;
    const K: c_int = 2560;
    const I: c_int = 640;
    const T_: c_int = 130;
    const TOPK = 10;
    const R: c_int = T_ * TOPK;
    const cases = [_]struct { fmt: Format, mode: [*:0]const u8, group: c_int, bits: c_int, gscale: bool }{
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4, .gscale = true },
        .{ .fmt = .nvfp4, .mode = "nvfp4", .group = 16, .bits = 4, .gscale = false },
        .{ .fmt = .mxfp8, .mode = "mxfp8", .group = 32, .bits = 8, .gscale = true },
    };
    for (cases, 0..) |cs, ci| {
        var arrs: [16]mlx.mlx_array = undefined;
        var na: usize = 0;
        defer for (arrs[0..na]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const keep = struct {
            fn f(list: *[16]mlx.mlx_array, n: *usize, a: mlx.mlx_array) mlx.mlx_array {
                list[n.*] = a;
                n.* += 1;
                return a;
            }
        }.f;
        var key = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&key, 31 + ci));
        _ = keep(&arrs, &na, key);
        var banks: [2][2]mlx.mlx_array = undefined;
        for (&banks, 0..) |*bank, bi| {
            var wf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wf);
            var k2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k2);
            try mlx.check(mlx.mlx_random_key(&k2, 100 + ci * 2 + bi));
            try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ E, I, K }, 3, .bfloat16, 0, 0.05, k2, s));
            var q = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(q);
            try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, .{}, s));
            for (bank, 0..) |*a, j| {
                a.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(a, q, j));
                _ = keep(&arrs, &na, a.*);
            }
        }
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ T_, K }, 2, .bfloat16, 0, 1, key, s));
        _ = keep(&arrs, &na, x);
        var gs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_uniform(&gs, mlx.mlx_array_new_float(0.5), mlx.mlx_array_new_float(2), &[_]c_int{E}, 1, .bfloat16, key, s));
        _ = keep(&arrs, &na, gs);
        // Routed rows sorted by expert, each with its token.
        const ids = try sortedIds(testing.allocator, @intCast(R), E, ci + 5);
        defer testing.allocator.free(ids);
        const toks = try testing.allocator.alloc(u32, @intCast(R));
        defer testing.allocator.free(toks);
        for (toks, 0..) |*v, i| v.* = @intCast((i * 37 + 11) % @as(usize, @intCast(T_)));
        const inds = keep(&arrs, &na, mlx.mlx_array_new_data(ids.ptr, &[_]c_int{R}, 1, .uint32));
        const rows = keep(&arrs, &na, mlx.mlx_array_new_data(toks.ptr, &[_]c_int{R}, 1, .uint32));

        const before = gu_served;
        const got = (try gateUp(s, x, rows, inds, banks[0][0], banks[0][1], banks[1][0], banks[1][1], if (cs.gscale) gs else null, cs.fmt)) orelse return error.Declined;
        _ = keep(&arrs, &na, got);
        try testing.expectEqual(before + 1, gu_served);

        // The chain: gather the token rows, each projection's sorted gather, the gate scale, silu(g) * u.
        var xr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&xr, x, rows, 0, s));
        _ = keep(&arrs, &na, xr);
        var x3 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&x3, xr, &[_]c_int{ R, 1, K }, 3, s));
        _ = keep(&arrs, &na, x3);
        var proj: [2]mlx.mlx_array = undefined;
        for (&proj, 0..) |*pj, bi| {
            var o3 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o3);
            try mlx.check(mlx.mlx_gather_qmm(&o3, x3, banks[bi][0], banks[bi][1], .{}, .{}, inds, true, mlx.mlx_optional_int.some(cs.group), mlx.mlx_optional_int.some(cs.bits), cs.mode, true, s));
            pj.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(pj, o3, &[_]c_int{ R, I }, 2, s));
        }
        defer for (proj) |pj| {
            _ = mlx.mlx_array_free(pj);
        };
        var g = proj[0];
        var gscaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gscaled);
        if (cs.gscale) {
            var picked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(picked);
            try mlx.check(mlx.mlx_take_axis(&picked, gs, inds, 0, s));
            var col = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(col);
            try mlx.check(mlx.mlx_expand_dims(&col, picked, -1, s));
            try mlx.check(mlx.mlx_multiply(&gscaled, g, col, s));
            g = gscaled;
        }
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_multiply(&sl, g, sig, s));
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_multiply(&ref, sl, proj[1], s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        if (!same) std.debug.print("gate_up {s} gscale={}: differs from the chain\n", .{ cs.mode, cs.gscale });
        try testing.expect(same);
    }
}
