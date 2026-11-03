//! sdk/blocks/gdn/kernels.
//! Owns gdnKernelSource, GdnOpsResult, gdnRecurOps, GDN_CUDA_HEADER, gdnCudaSource, getGdnCudaKernel, and 41 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("decode.zig");
const testing = std.testing;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const evalToF32 = @import("../../testing/parity.zig").evalToF32;
const hostF32 = @import("../../testing/parity.zig").hostF32;
const maxAbsDiff = @import("../../testing/parity.zig").maxAbsDiff;
const rmsVsTruth = @import("../../testing/parity.zig").rmsVsTruth;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const attn256RandBf16Scaled = @import("../attention/attn256.zig").attn256RandBf16Scaled;
const gdnChainGateDtype = @import("gate.zig").gdnChainGateDtype;
const gdnGateChain = @import("gate.zig").gdnGateChain;

// ── GatedDeltaNet fused Metal kernel ──
// Ported from mlx-lm/models/gated_delta.py: `_make_gated_delta_kernel(has_mask=False, vectorized=False)`.
// Processes the entire T-step delta recurrence in a single kernel dispatch, eliminating
// the per-token kernel-launch overhead that otherwise caps prefill at ~330 tok/s on
// Qwen 3.5/3.6 MoE. Template args (Dk, Dv, Hk, Hv) specialize the kernel; inputs carry
// the runtime shapes. State math runs in float32 for numerical stability regardless
// of the input/state storage dtype.
// The recurrence exists ONCE, specialized on two independent axes:
//
//   `vectorized` — where the forget gate lives. qwen3.5's GatedDeltaNet has one
//     scalar per (token, value head), so `g` is [B,T,Hv]. KDA (Kimi Delta
//     Attention, bailing_hybrid) gates per KEY CHANNEL, so `g` is [B,T,Hv,Dk]
//     and each state element decays by its own factor. Mirrors mlx-lm's
//     `_make_gated_delta_kernel(vectorized=…)`, which parameterizes the same
//     three lines.
//
//   `capture_seq` — spec-decode support, used ONLY on the verify forward (small
//     T). Besides the FINAL state (`state_out`, written from registers like the
//     stock kernel) it records the state after every INTERMEDIATE timestep into
//     `state_seq` ([T, B, Hv, Dv, Dk]), so partial-accept rollback restores the
//     accepted position by SLICING — no re-forward of the accepted prefix (the
//     costly part on a 48-layer trunk). Decode and prefill never pay for it.
//     `seq_stride` = (B*Hv)*Dv*Dk = the per-timestep element stride into
//     `state_seq` (passed as a scalar, mirroring `T`).
//     Capture-tail trim: state_seq[T-1] is deliberately NEVER written — a
//     partial accept reads index `accepted` <= T-2 (index T-1 would be a full
//     accept, which keeps `state_out` via the normal flow), so the last row was
//     a redundant global write + forced the engine's final state to be a slice
//     VIEW pinning the whole [T,...] capture buffer across rounds.
pub fn gdnKernelSource(comptime vectorized: bool, comptime capture_seq: bool) [:0]const u8 {
    const g_setup = if (vectorized)
        "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
    else
        "auto g_ = g + b_idx * T * Hv;";
    const g_access = if (vectorized) "g_[s_idx]" else "g_[hv_idx]";
    const g_advance = if (vectorized) "  g_ += Hv * Dk;" else "  g_ += Hv;";
    const seq_base = if (capture_seq)
        "auto seq_base = (n * Dv + dv_idx) * Dk;\n"
    else
        "";
    const seq_write = if (capture_seq)
        kernelSource("kernels/gdn/gated_delta_seq_write.metal")
    else
        "";
    return std.fmt.comptimePrint(kernelSource("kernels/gdn/gated_delta.fmt.metal"), .{ seq_base, g_setup, g_access, seq_write, g_advance });
}

const GdnOpsResult = struct { y: mlx.mlx_array, state: mlx.mlx_array, state_seq: mlx.mlx_array };

/// The `gated_delta_step` recurrence as MLX ops, one step per token, for GPUs
/// that run no Metal kernels (CUDA). Same contract and f32 state math as the
/// kernel; `state_seq` ([T,B,Hv,Dv,Dk], every step) only when `capture`.
pub fn gdnRecurOps(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, vectorized: bool, capture: bool, out_dtype: mlx.mlx_dtype) !GdnOpsResult {
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    const b = qs[0];
    const t_len = qs[1];
    const hk = qs[2];
    const dk = qs[3];
    const hv = vs[2];
    const dv = vs[3];
    const state_dtype = mlx.mlx_array_dtype(state_in);

    // f32, k/q heads repeated onto the value heads, shaped for per-step matmuls.
    const Prep = struct {
        fn run(st: mlx.mlx_stream, a: mlx.mlx_array, rep: c_int, shape: []const c_int) !mlx.mlx_array {
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, a, .float32, st));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            if (rep > 1) try mlx.check(mlx.mlx_repeat_axis(&r, f, rep, 2, st)) else try mlx.check(mlx.mlx_copy(&r, f, st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&out, r, shape.ptr, shape.len, st));
            return out;
        }
        fn step(st: mlx.mlx_stream, a: mlx.mlx_array, t: c_int) !mlx.mlx_array {
            const sh = mlx.getShape(a);
            var start = [5]c_int{ 0, 0, 0, 0, 0 };
            var stop: [5]c_int = undefined;
            const strides = [5]c_int{ 1, 1, 1, 1, 1 };
            for (sh, 0..) |d, i| stop[i] = d;
            start[1] = t;
            stop[1] = t + 1;
            var sl = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sl);
            try mlx.check(mlx.mlx_slice(&sl, a, &start, sh.len, &stop, sh.len, &strides, sh.len, st));
            var shape: [4]c_int = undefined;
            shape[0] = sh[0];
            for (sh[2..], 1..) |d, i| shape[i] = d;
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&out, sl, &shape, sh.len - 1, st));
            return out;
        }
    };
    const rep = @divExact(hv, hk);
    const q_col = try Prep.run(s, q, rep, &.{ b, t_len, hv, dk, 1 });
    defer _ = mlx.mlx_array_free(q_col);
    const k_col = try Prep.run(s, k, rep, &.{ b, t_len, hv, dk, 1 });
    defer _ = mlx.mlx_array_free(k_col);
    const k_row = try Prep.run(s, k, rep, &.{ b, t_len, hv, 1, dk });
    defer _ = mlx.mlx_array_free(k_row);
    const v_col = try Prep.run(s, v, 1, &.{ b, t_len, hv, dv, 1 });
    defer _ = mlx.mlx_array_free(v_col);
    const beta_b = try Prep.run(s, beta, 1, &.{ b, t_len, hv, 1, 1 });
    defer _ = mlx.mlx_array_free(beta_b);
    const g_b = try Prep.run(s, g, 1, if (vectorized) &.{ b, t_len, hv, 1, dk } else &.{ b, t_len, hv, 1, 1 });
    defer _ = mlx.mlx_array_free(g_b);

    var state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&state, state_in, .float32, s));
    defer _ = mlx.mlx_array_free(state);
    const n: usize = @intCast(t_len);
    const ys = try std.heap.c_allocator.alloc(mlx.mlx_array, n);
    defer std.heap.c_allocator.free(ys);
    var n_ys: usize = 0;
    defer for (ys[0..n_ys]) |y| {
        _ = mlx.mlx_array_free(y);
    };
    const seq = try std.heap.c_allocator.alloc(mlx.mlx_array, if (capture) n else 0);
    defer std.heap.c_allocator.free(seq);
    var n_seq: usize = 0;
    defer for (seq[0..n_seq]) |x| {
        _ = mlx.mlx_array_free(x);
    };

    var t: c_int = 0;
    while (t < t_len) : (t += 1) {
        const gt = try Prep.step(s, g_b, t);
        defer _ = mlx.mlx_array_free(gt);
        const kc = try Prep.step(s, k_col, t);
        defer _ = mlx.mlx_array_free(kc);
        const kr = try Prep.step(s, k_row, t);
        defer _ = mlx.mlx_array_free(kr);
        const qc = try Prep.step(s, q_col, t);
        defer _ = mlx.mlx_array_free(qc);
        const vc = try Prep.step(s, v_col, t);
        defer _ = mlx.mlx_array_free(vc);
        const bt = try Prep.step(s, beta_b, t);
        defer _ = mlx.mlx_array_free(bt);

        var decayed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(decayed);
        try mlx.check(mlx.mlx_multiply(&decayed, state, gt, s));
        var kv_mem = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kv_mem);
        try mlx.check(mlx.mlx_matmul(&kv_mem, decayed, kc, s));
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, vc, kv_mem, s));
        var delta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(delta);
        try mlx.check(mlx.mlx_multiply(&delta, diff, bt, s));
        var upd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upd);
        try mlx.check(mlx.mlx_matmul(&upd, delta, kr, s));
        var next = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&next, decayed, upd, s));
        _ = mlx.mlx_array_free(state);
        state = next;

        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&y, state, qc, s));
        ys[n_ys] = y;
        n_ys += 1;
        if (capture) {
            var c = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&c, state, state_dtype, s));
            seq[n_seq] = c;
            n_seq += 1;
        }
    }

    var y_stack = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_stack);
    {
        const vec = mlx.mlx_vector_array_new_data(ys.ptr, n_ys);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_stack_axis(&y_stack, vec, 1, s));
    }
    var y_shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_shaped);
    const y_shape = [_]c_int{ b, t_len, hv, dv };
    try mlx.check(mlx.mlx_reshape(&y_shaped, y_stack, &y_shape, 4, s));
    var out: GdnOpsResult = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = .{ .ctx = null } };
    errdefer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.state);
    }
    try mlx.check(mlx.mlx_astype(&out.y, y_shaped, out_dtype, s));
    try mlx.check(mlx.mlx_astype(&out.state, state, state_dtype, s));
    if (capture) {
        const vec = mlx.mlx_vector_array_new_data(seq.ptr, n_seq);
        defer _ = mlx.mlx_vector_array_free(vec);
        out.state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_stack_axis(&out.state_seq, vec, 0, s));
    }
    return out;
}

const GDN_CUDA_HEADER = kernelSource("kernels/gdn/gated_delta_header.cu");

/// `gdnKernelSource` for MLX's CUDA backend: one warp per (head, RPW Dv rows), the Dk lanes reduced
/// by shuffle. A row's arithmetic does not depend on RPW; rows of one head share the step's loads.
/// Scalars arrive as pointers (`T_arr`).
fn gdnCudaSource(comptime vectorized: bool, comptime capture_seq: bool) [:0]const u8 {
    // Step t+1's inputs load while step t computes; each converts to float as it did in place.
    const ng = if (vectorized) "n_per_t" else "1";
    const g_load = if (vectorized) "gn[i] = static_cast<float>(g_[(long)tt * Hv * Dk + s_idx]);" else "if (i == 0) gn[0] = static_cast<float>(g_[(long)tt * Hv + hv_idx]);";
    const g_use = if (vectorized) "gc[i]" else "gc[0]";
    const g_setup = if (vectorized) "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;" else "auto g_ = g + b_idx * T * Hv;";
    const seq_write = if (capture_seq)
        kernelSource("kernels/gdn/gated_delta_seq_write.cu")
    else
        "";
    return std.fmt.comptimePrint(kernelSource("kernels/gdn/gated_delta.fmt.cu"), .{ g_setup, ng, g_load, ng, ng, g_use, seq_write });
}

fn getGdnCudaKernel(comptime vectorized: bool, comptime capture_seq: bool) !mlx.mlx_fast_cuda_kernel {
    const Cache = struct {
        const is_vectorized = vectorized;
        const is_capture_seq = capture_seq;
        var kernel: ?mlx.mlx_fast_cuda_kernel = null;
    };
    if (Cache.kernel) |k| return k;
    const input_names = if (capture_seq)
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T_arr", "seq_stride_arr" }
    else
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T_arr" };
    const output_names = if (capture_seq)
        [_][*:0]const u8{ "y", "state_seq", "state_out" }
    else
        [_][*:0]const u8{ "y", "state_out" };
    const name = "gated_delta_step_cuda" ++ (if (capture_seq) "_seq" else "") ++ (if (vectorized) "_vec" else "");
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, gdnCudaSource(vectorized, capture_seq), GDN_CUDA_HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    Cache.kernel = k;
    return k;
}

/// Dv rows per warp: a prefill-length recurrence shares each step's loads over 16 rows (fastest of
/// 1-16 on GB10); a decode step keeps one row per warp for the parallelism.
fn gdnCudaRowsPerWarp(t_len: c_int, dv: c_int) c_int {
    return if (t_len >= 64 and @rem(dv, 64) == 0) 16 else 1;
}

/// Test seam: rows per warp for `gdnRecurCuda` (null = `gdnCudaRowsPerWarp`).
var gdn_cuda_rows_per_warp: ?c_int = null;

/// The GDN recurrence on MLX's CUDA backend, same contract as `gdnRecurOps`;
/// null when not on CUDA or Dk is not a warp multiple.
pub fn gdnRecurCuda(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, vectorized: bool, capture: bool, out_dtype: mlx.mlx_dtype) !?GdnOpsResult {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    const b = qs[0];
    const t_len = qs[1];
    const hk = qs[2];
    const dk = qs[3];
    const hv = vs[2];
    const dv = vs[3];
    if (@rem(dk, 32) != 0 or @rem(hv, hk) != 0) return null;
    const state_dtype = mlx.mlx_array_dtype(state_in);

    const config = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(config);
    const y_shape = [_]c_int{ b, t_len, hv, dv };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &y_shape, 4, out_dtype));
    if (capture) {
        const seq_shape = [_]c_int{ t_len, b, hv, dv, dk };
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &seq_shape, 5, state_dtype));
    }
    const state_shape = [_]c_int{ b, hv, dv, dk };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &state_shape, 4, state_dtype));
    const rpw = gdn_cuda_rows_per_warp orelse gdnCudaRowsPerWarp(t_len, dv);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(config, 32, @divExact(dv, rpw), b * hv));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "RPW", rpw));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "StT", state_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Hv", hv));

    const t_arr = mlx.mlx_array_new_int(t_len);
    defer _ = mlx.mlx_array_free(t_arr);
    const stride_arr = mlx.mlx_array_new_int(b * hv * dv * dk);
    defer _ = mlx.mlx_array_free(stride_arr);
    const ins = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, t_arr, stride_arr };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, if (capture) 8 else 7);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    const kernel = switch (vectorized) {
        inline else => |vec| switch (capture) {
            inline else => |cap| try getGdnCudaKernel(vec, cap),
        },
    };
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&outs, kernel, in_vec, config, s));
    var r: GdnOpsResult = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = .{ .ctx = null } };
    try mlx.check(mlx.mlx_vector_array_get(&r.y, outs, 0));
    if (capture) {
        r.state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&r.state_seq, outs, 1));
    }
    try mlx.check(mlx.mlx_vector_array_get(&r.state, outs, if (capture) 2 else 1));
    return r;
}

/// One compiled kernel per (vectorized, capture_seq) pair. Each comptime
/// instantiation gets its own cache slot; the Metal name must differ per
/// variant or MLX hands back the wrong compiled function.
fn getGdnKernelFor(comptime vectorized: bool, comptime capture_seq: bool) !mlx.mlx_fast_metal_kernel {
    // The two `const`s are load-bearing: a struct declared inside a function
    // that captures NOTHING from the comptime parameters is memoized to ONE
    // type across every instantiation, so all four variants would share a
    // single cache slot and the first kernel compiled would be handed back for
    // the rest (a seq kernel answering a non-seq call = wrong output count).
    const Cache = struct {
        const is_vectorized = vectorized;
        const is_capture_seq = capture_seq;
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    if (Cache.kernel) |k| return k;
    const input_names = if (capture_seq)
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" }
    else
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = if (capture_seq)
        [_][*:0]const u8{ "y", "state_seq", "state_out" }
    else
        [_][*:0]const u8{ "y", "state_out" };
    const name = "gated_delta_step" ++
        (if (capture_seq) "_seq" else "") ++
        (if (vectorized) "_vec" else "");
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        gdnKernelSource(vectorized, capture_seq),
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = kernel;
    return kernel;
}

pub fn getGdnKernel(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, false) else getGdnKernelFor(false, false);
}

pub fn getGdnKernelSeq(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, true) else getGdnKernelFor(false, true);
}

// ── GatedDeltaNet blocked-seq PREFILL kernel ──
// Port of oMLX's `gated_delta_blocked_seq` (custom_kernels/qwen35_prefill/
// gdn.py, Apache-2.0, oMLX by jundot — keep this provenance chain). Same
// EXACT recurrence as GDN_KERNEL_SOURCE — no chunked/WY reformulation —
// restructured for Apple-GPU efficiency: k/q/v staged into threadgroup
// memory in TB-token blocks with coalesced cooperative loads (the stock
// kernel re-reads k/q from device once per (Dv/4)-slice threadgroup => 32x
// redundant traffic, ~13 GB per 16K-token layer), state register-resident
// as (dv row, 16-wide d segment) fragments, and the k·state / q·state
// contractions reduced across the 8 segment-threads of a dv row via
// simd_shuffle_down — no threadgroup barriers in the hot token loop. Their
// measurement: 14.9 ms vs 29.7 ms per layer @16K = ~2x over the stock
// kernel shape; 48 of the 27B's 64 layers are GDN.
// Geometry contract (checked by gdnBlockedEligible): Dk == 128 exactly
// (8 threads x 16-wide fragments), Dv % 32 == 0 (DB=32 dv rows per
// threadgroup). Anything else — and decode (T==1), PLD/MTP verify, and the
// per-position-state capture path — stays on the stock kernels.
// Block size 32 for bf16 (Metal's 32 KiB threadgroup limit governs,
// and it is a function of the INPUT WIDTH, not a constant: fp32 activations
// stage twice the bytes and clamp to 16 via gdnBlockTFor. GDN inputs are NOT
// always bf16 — an f16 checkpoint promotes its activations to f32).

/// Test seam: forces the blocked-prefill route on/off.
pub var gdn_blocked_override: ?bool = null;

pub fn gdnBlockedEnabled() bool {
    return gdn_blocked_override orelse true;
}

/// Prefill-width floor for the blocked kernel (mirrors oMLX's OMLX_GDN_MIN_T
/// default). Below this the per-token stock kernel wins on launch overhead.
pub const GDN_BLOCKED_MIN_T: c_int = 64;

/// Pure routing predicate: geometry + width gate for the blocked-seq kernel.
pub fn gdnBlockedEligible(seq_len: c_int, dk: c_int, dv: c_int, num_k_heads: c_int, num_v_heads: c_int) bool {
    if (seq_len < GDN_BLOCKED_MIN_T) return false;
    if (dk != 128) return false;
    if (dv < 32 or @rem(dv, 32) != 0) return false;
    if (num_k_heads <= 0 or @rem(num_v_heads, num_k_heads) != 0) return false;
    return true;
}

/// Metal's threadgroup memory budget per threadgroup on Apple GPUs. The
/// blocked kernel stages k/q/v in threadgroup memory, so this is a HARD cap on
/// TB x input width — exceed it and the kernel does not compile.
const METAL_TG_BYTES: usize = 32 * 1024;

/// Threadgroup bytes the blocked kernel stages per TB-token block:
/// `k_s`+`q_s` [TB][Dk+8] and `v_s` [TB][DB+8] at the input width, plus the
/// `g_s`/`b_s` [TB] f32 rows. Mirrors the declarations in
/// GDN_KERNEL_BLOCKED_BODY — change one, change the other.
pub fn gdnBlockedTgBytes(tb: u32, dk: c_int, in_elem_size: usize) usize {
    const DB: usize = 32; // dv rows per threadgroup
    const t: usize = tb;
    const k: usize = @intCast(dk);
    return t * (2 * (k + 8) * in_elem_size + (DB + 8) * in_elem_size + 2 * @sizeOf(f32));
}

/// Largest supported block size <= `requested` whose staging fits the budget
/// at this input width, or null when even the smallest does not (decline the
/// blocked route and stay on the stock kernel).
///
/// The width is NOT a constant: bf16 and f16 checkpoints stage 2 bytes and
/// keep the requested size, but an f16 checkpoint promotes its activations to
/// f32 (f16 (x) f32-scalar -> f32), and at 4 bytes TB=32 needs 40 KiB. Block
/// size and input dtype are therefore ONE decision — picking them apart is how
/// a valid-looking config produces a kernel that cannot compile.
pub fn gdnBlockTFor(requested: u32, dk: c_int, in_elem_size: usize) ?u32 {
    var best: ?u32 = null;
    for (GDN_BLOCKED_TBS) |cand| {
        if (cand > requested) continue;
        if (gdnBlockedTgBytes(cand, dk, in_elem_size) > METAL_TG_BYTES) continue;
        if (best == null or cand > best.?) best = cand;
    }
    return best;
}

const GDN_BLOCKED_TBS = [_]u32{ 16, 32, 48 };
var gdn_blocked_cached: [GDN_BLOCKED_TBS.len]?mlx.mlx_fast_metal_kernel = @splat(null);
/// The blocked kernel's time block.
pub const GDN_BLOCK_T: u32 = 32;

// Body of the blocked-seq kernel; the `constexpr int TB = N;` line is
// prepended per block size by gdnBlockedSource (one cached kernel per TB,
// DISTINCT names — two specializations sharing a name bind the wrong binary).
// Faithful transcription of oMLX's _KERNEL_S_SRC with one generalization:
// state loads/stores go through vec<StT,4> (theirs hard-codes float4 — their
// state is fp32, ours rides the ssm_state buffer's own dtype exactly like the
// stock kernel, so cross-chunk behavior is unchanged).
const GDN_KERNEL_BLOCKED_BODY = kernelSource("kernels/gdn/gated_delta_blocked_body.metal");

fn gdnBlockedSource(comptime tb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("constexpr int TB = {d};\n", .{tb}) ++ GDN_KERNEL_BLOCKED_BODY;
}

fn getGdnKernelBlocked(tb: u32) !mlx.mlx_fast_metal_kernel {
    inline for (GDN_BLOCKED_TBS, 0..) |cand, i| {
        if (cand == tb) {
            if (gdn_blocked_cached[i]) |k| return k;
            const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
            const output_names = [_][*:0]const u8{ "y", "state_out" };
            const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
            defer _ = mlx.mlx_vector_string_free(in_vec);
            const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
            defer _ = mlx.mlx_vector_string_free(out_vec);
            const kernel = mlx.mlx_fast_metal_kernel_new(
                std.fmt.comptimePrint("gated_delta_blocked_tb{d}", .{cand}),
                in_vec,
                out_vec,
                gdnBlockedSource(cand),
                "",
                true,
                false,
            );
            if (kernel.ctx == null) return error.MetalKernelCompileFailed;
            gdn_blocked_cached[i] = kernel;
            return kernel;
        }
    }
    return error.UnsupportedGdnBlockT;
}

/// Which recurrence kernel a GDN call runs: the per-token stock kernel, the blocked-seq
/// prefill kernel, or oMLX's software-pipelined one, which takes every call the blocked
/// kernel would.
pub const GdnRoute = enum { stock, blocked, pipelined };

var gdn_pipelined_cached: ?mlx.mlx_fast_metal_kernel = null;
pub var gdn_pipelined_override: ?bool = null; // test seam

/// Archs whose prefill was A/B'd against the blocked kernel on an M5 Ultra: Flash Next
/// (`qwen4_exp`, +2.6%) and dense Qwen3.8-27B (`qwen3_5`, +1.1 to 1.5%), both 48 value heads.
/// The kernel is issue- and latency-bound, so an unmeasured arch or chip keeps the blocked
/// kernel until someone measures it.
const GDN_PIPELINED_MEASURED = [_][]const u8{ "qwen4_exp", "qwen3_5" };

/// Whether a GDN prefill takes the pipelined kernel. It sums the recurrence's two 128-wide
/// dots in another fp32 order than the blocked one, so greedy text can move.
/// Only a measured arch on an M5-class GPU takes it.
pub fn gdnPipelinedFor(model_type: []const u8, nax: bool) bool {
    if (!nax) return false;
    for (GDN_PIPELINED_MEASURED) |t| {
        if (std.mem.eql(u8, t, model_type)) return true;
    }
    return false;
}

/// Decided per call, since the registry serves several model types in one process.
pub fn gdnPipelinedEnabled(model_type: []const u8) bool {
    if (gdn_pipelined_override) |v| return v;
    return gdnPipelinedFor(model_type, naxAvailable());
}

fn getGdnKernelPipelined() !mlx.mlx_fast_metal_kernel {
    if (gdn_pipelined_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = [_][*:0]const u8{ "y", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("gated_delta_pipelined_tb12", in_vec, out_vec, kernelSource("kernels/gdn/gdn_pipelined.metal"), "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_pipelined_cached = kernel;
    return kernel;
}

pub fn gdnRouteKernel(route: GdnRoute, tb: u32, vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return switch (route) {
        .stock => getGdnKernel(vector_gate),
        .blocked => getGdnKernelBlocked(tb),
        .pipelined => getGdnKernelPipelined(),
    };
}

pub fn gdnSetGrid(config: mlx.mlx_fast_metal_kernel_config, route: GdnRoute, dv: c_int, hv: c_int, batch: c_int) !void {
    switch (route) {
        // (32, Dv, B*Hv) threads in (32, 4, 1) groups, as mlx-lm.
        .stock => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
        },
        // 256 threads own 32 value rows.
        .blocked => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256 * @divExact(dv, 32), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        },
        // 128 threads own 16 value rows, 8 lanes each; Dv is a multiple of 32 on this route.
        .pipelined => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128 * @divExact(dv, 16), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        },
    }
}

// ── tests ──

test "gdnRecurCuda matches the op-chain recurrence and is no worse against f64 truth (scalar and per-channel gates, with capture)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const T: c_int = 19;
    const Hk: c_int = 2;
    const Hv: c_int = 4;
    const Dk: c_int = 128;
    const Dv: c_int = 64;
    const R = struct {
        fn arr(st: mlx.mlx_stream, seed: u64, shape: []const c_int, scale: f32, sig: bool) !mlx.mlx_array {
            var key = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(key);
            try mlx.check(mlx.mlx_random_key(&key, seed));
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&a, shape.ptr, shape.len, .float32, 0, scale, key, st));
            if (sig) {
                var b = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_sigmoid(&b, a, st));
                _ = mlx.mlx_array_free(a);
                return b;
            }
            return a;
        }
    };
    const a = std.testing.allocator;
    const ndk: usize = @intCast(Dk);
    const ndv: usize = @intCast(Dv);
    const nt: usize = @intCast(T);
    const rep: usize = @intCast(@divExact(Hv, Hk));
    // Squared errors against the f64 truth, pooled over seeds: kernel then chain, y then state.
    var sse = [4]f64{ 0, 0, 0, 0 };
    for (0..8) |seed| {
        const base: u64 = 16 * seed;
        const q = try R.arr(s, base + 1, &.{ B, T, Hk, Dk }, 0.1, false);
        defer _ = mlx.mlx_array_free(q);
        const k = try R.arr(s, base + 2, &.{ B, T, Hk, Dk }, 0.1, false);
        defer _ = mlx.mlx_array_free(k);
        const v = try R.arr(s, base + 3, &.{ B, T, Hv, Dv }, 1, false);
        defer _ = mlx.mlx_array_free(v);
        const beta = try R.arr(s, base + 4, &.{ B, T, Hv }, 1, true);
        defer _ = mlx.mlx_array_free(beta);
        const state = try R.arr(s, base + 5, &.{ B, Hv, Dv, Dk }, 0.1, false);
        defer _ = mlx.mlx_array_free(state);
        for ([_]bool{ false, true }) |vec| {
            const g = try R.arr(s, base + 6, if (vec) &[_]c_int{ B, T, Hv, Dk } else &[_]c_int{ B, T, Hv }, 1, true);
            defer _ = mlx.mlx_array_free(g);
            for ([_]bool{ false, true }) |cap| {
                const got = (try gdnRecurCuda(s, q, k, v, g, beta, state, vec, cap, .float32)) orelse return error.Declined;
                defer for ([_]mlx.mlx_array{ got.y, got.state, got.state_seq }) |x| if (x.ctx != null) {
                    _ = mlx.mlx_array_free(x);
                };
                const want = try gdnRecurOps(s, q, k, v, g, beta, state, vec, cap, .float32);
                defer for ([_]mlx.mlx_array{ want.y, want.state, want.state_seq }) |x| if (x.ctx != null) {
                    _ = mlx.mlx_array_free(x);
                };
                try std.testing.expect(try attn256MaxDiff(got.y, want.y, s) < 1e-4);
                try std.testing.expect(try attn256MaxDiff(got.state, want.state, s) < 1e-4);
                if (cap) try std.testing.expect(try attn256MaxDiff(got.state_seq, want.state_seq, s) < 1e-4);
                if (cap) continue; // the captured run's y and state are the plain run's

                const qh = try hostF32(q);
                const kh = try hostF32(k);
                const vh = try hostF32(v);
                const gh = try hostF32(g);
                const bh = try hostF32(beta);
                const sh = try hostF32(state);
                const y_t = try a.alloc(f64, vh.len);
                defer a.free(y_t);
                const s_t = try a.alloc(f64, sh.len);
                defer a.free(s_t);
                for (sh, s_t) |x, *o| o.* = x;
                for (0..@intCast(B)) |bi| for (0..@intCast(Hv)) |h| {
                    const st = s_t[(bi * @as(usize, @intCast(Hv)) + h) * ndv * ndk ..][0 .. ndv * ndk];
                    for (0..nt) |t| {
                        const hv_i = (bi * nt + t) * @as(usize, @intCast(Hv)) + h;
                        const kq = ((bi * nt + t) * @as(usize, @intCast(Hk)) + h / rep) * ndk;
                        for (0..ndv) |i| {
                            var kv: f64 = 0;
                            for (0..ndk) |j| {
                                st[i * ndk + j] *= if (vec) gh[hv_i * ndk + j] else gh[hv_i];
                                kv += st[i * ndk + j] * kh[kq + j];
                            }
                            const delta = (@as(f64, vh[hv_i * ndv + i]) - kv) * bh[hv_i];
                            var y: f64 = 0;
                            for (0..ndk) |j| {
                                st[i * ndk + j] += delta * kh[kq + j];
                                y += st[i * ndk + j] * qh[kq + j];
                            }
                            y_t[hv_i * ndv + i] = y;
                        }
                    }
                };
                for ([_]mlx.mlx_array{ got.y, want.y, got.state, want.state }, 0..) |x, i| {
                    const truth = if (i < 2) y_t else s_t;
                    const e = try rmsVsTruth(x, truth);
                    sse[i] += e * e * @as(f64, @floatFromInt(truth.len));
                }
            }
        }
    }
    std.debug.print("[gdn cuda vs f64, 8 seeds] y kernel/chain = {d:.4}, state kernel/chain = {d:.4}\n", .{ @sqrt(sse[0] / sse[1]), @sqrt(sse[2] / sse[3]) });
    try std.testing.expect(sse[0] <= sse[1] and sse[2] <= sse[3]);
}

test "gdnRecurCuda: four and sixteen Dv rows per warp are bit-identical to one (scalar and per-channel gates, with capture)" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    defer gdn_cuda_rows_per_warp = null;
    const B: c_int = 1;
    const T: c_int = 77;
    const Hk: c_int = 2;
    const Hv: c_int = 4;
    const Dk: c_int = 128;
    const Dv: c_int = 128;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 6));
    var arrs: [6]mlx.mlx_array = undefined;
    const shapes = [_][]const c_int{ &.{ B, T, Hk, Dk }, &.{ B, T, Hk, Dk }, &.{ B, T, Hv, Dv }, &.{ B, T, Hv }, &.{ B, Hv, Dv, Dk }, &.{ B, T, Hv, Dk } };
    for (&arrs, shapes, 0..) |*a, sh, i| {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, .float32, 0, if (i < 2 or i == 4) 0.1 else 1, key, s));
        a.* = mlx.mlx_array_new();
        if (i == 3 or i == 5) try mlx.check(mlx.mlx_sigmoid(a, r, s)) else try mlx.check(mlx.mlx_astype(a, r, if (i == 4) .float32 else .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]bool{ false, true }) |vec| for ([_]bool{ false, true }) |cap| {
        const g = if (vec) arrs[5] else arrs[3];
        var runs: [3]GdnOpsResult = undefined;
        var n_runs: usize = 0;
        defer for (runs[0..n_runs]) |r| for ([_]mlx.mlx_array{ r.y, r.state, r.state_seq }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]c_int{ 1, 4, 16 }) |rpw| {
            gdn_cuda_rows_per_warp = rpw;
            runs[n_runs] = (try gdnRecurCuda(s, arrs[0], arrs[1], arrs[2], g, arrs[3], arrs[4], vec, cap, .bfloat16)) orelse return error.Declined;
            n_runs += 1;
        }
        for (runs[1..]) |r| {
            for ([_][2]mlx.mlx_array{ .{ runs[0].y, r.y }, .{ runs[0].state, r.state }, .{ runs[0].state_seq, r.state_seq } }) |pair| {
                if (pair[0].ctx == null) continue;
                var eq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(eq);
                try mlx.check(mlx.mlx_array_equal(&eq, pair[0], pair[1], false, s));
                try mlx.check(mlx.mlx_array_eval(eq));
                var same = false;
                try mlx.check(mlx.mlx_array_item_bool(&same, eq));
                try std.testing.expect(same);
            }
        }
    };
}

test "gdnGateChain matches g = exp(-exp(A_log) * softplus(a + dt_bias))" {
    const s = mlx.gpuStream();

    // Hv = 3 heads, B=1, S=1. Hand-computable fixtures:
    //   head 0: A_log=0, a=0, dt_bias=0 → g = exp(-1 * softplus(0)) = exp(-ln2) = 0.5
    //   head 1: A_log=ln2, a=0, dt_bias=0 → g = exp(-2 * ln2) = 0.25
    //   head 2: exp(A_log) * ln2 = 5e-4 → g = exp(-5e-4), which bf16 rounds to 1.0
    const a_log_data = [_]f32{ 0.0, @log(2.0), @log(5e-4 / std.math.ln2) };
    const hv_shape = [_]c_int{3};
    const A_log = mlx.mlx_array_new_data(&a_log_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(A_log);

    const a_data = [_]f32{ 0.0, 0.0, 0.0 };
    const a_shape = [_]c_int{ 1, 1, 3 };
    const a = mlx.mlx_array_new_data(&a_data, &a_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(a);

    const dt_data = [_]f32{ 0.0, 0.0, 0.0 };
    const dt_bias = mlx.mlx_array_new_data(&dt_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(dt_bias);

    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |g_dtype| {
        const g = try gdnGateChain(A_log, a, dt_bias, g_dtype, s);
        defer _ = mlx.mlx_array_free(g);
        try testing.expectEqual(g_dtype, mlx.mlx_array_dtype(g));

        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_astype(&g_f32, g, .float32, s));
        try mlx.check(mlx.mlx_array_eval(g_f32));

        const gd = mlx.mlx_array_data_float32(g_f32) orelse return error.InvalidDtype;
        // bf16 round-trip tolerance, or f32's
        const tol: f32 = if (g_dtype == .float32) 1e-6 else 5e-3;
        try testing.expect(@abs(gd[0] - 0.5) < tol);
        try testing.expect(@abs(gd[1] - 0.25) < tol);
        if (g_dtype == .float32) try testing.expect(@abs(gd[2] - @exp(@as(f32, -5e-4))) < tol);
    }
}

// Helper for the GDN seq-kernel parity test below: run the recurrence via the
// requested kernel variant and return the FINAL state [B,Hv,Dv,Dk] as f32.
fn gdnTestRun(comptime seq: bool, comptime vec: bool, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    if (seq) {
        const ss_shape = [_]c_int{ T, B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ss_shape, 5, .bfloat16));
        // Final state is its own output (written from registers); the seq
        // buffer's last row is deliberately UNWRITTEN (capture-tail trim).
        const so_shape = [_]c_int{ B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    } else {
        const so_shape = [_]c_int{ B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, Dv, B * Hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    if (seq) {
        const seq_stride = mlx.mlx_array_new_int(B * Hv * Dv * Dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar, seq_stride };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);
        const kern = try getGdnKernelSeq(vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
        // Final state comes from the dedicated state_out output — after the
        // capture-tail trim, state_seq[T-1] is never written.
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 2));
        return out;
    } else {
        const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);
        const kern = try getGdnKernel(vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 1));
        return out;
    }
}

test "GDN seq-kernel final state matches single-state kernel (PLD capture parity)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const T: c_int = 4;
    const Hk: c_int = 1;
    const Hv: c_int = 2;
    const Dk: c_int = 128; // n_per_t = Dk/32 = 4 (matches real GatedDeltaNet head dim)
    const Dv: c_int = 4;

    // Deterministic pseudo-random inputs in [-0.5, 0.5].
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);
    const qd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(qd);
    const kd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(kd);
    const vd = try testing.allocator.alloc(f32, vn);
    defer testing.allocator.free(vd);
    const gd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(gd);
    const bd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(bd);
    const sd = try testing.allocator.alloc(f32, sn);
    defer testing.allocator.free(sd);
    for (qd) |*x| x.* = rnd.float(f32) - 0.5;
    for (kd) |*x| x.* = rnd.float(f32) - 0.5;
    for (vd) |*x| x.* = rnd.float(f32) - 0.5;
    for (gd) |*x| x.* = 0.5 + 0.4 * rnd.float(f32); // decay in (0.5,0.9)
    for (bd) |*x| x.* = rnd.float(f32);
    for (sd) |*x| x.* = rnd.float(f32) - 0.5;

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    // Cast inputs to bf16 (kernel expects InT/StT = bf16).
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_astype(&q, q32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&v, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&st, st32, .bfloat16, s));

    const state_a = try gdnTestRun(false, false, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_a);
    const state_b = try gdnTestRun(true, false, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_b);

    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var bb32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bb32);
    try mlx.check(mlx.mlx_astype(&a32, state_a, .float32, s));
    try mlx.check(mlx.mlx_astype(&bb32, state_b, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(bb32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd2 = mlx.mlx_array_data_float32(bb32) orelse return error.InvalidDtype;
    var max_diff: f32 = 0;
    for (0..sn) |i| max_diff = @max(max_diff, @abs(ad[i] - bd2[i]));
    // Both write the same bf16-cast state — expect exact (allow tiny slack).
    try testing.expect(max_diff < 1e-3);

    // Intermediate-position parity: the state captured at position `accepted`
    // of a length-T seq run must equal a fresh normal-kernel run over just the
    // first (accepted+1) tokens — this is exactly what partial-accept rollback
    // relies on. Check accepted = 1 (run length 2).
    const acc_run: c_int = 2; // accepted+1 = 2 tokens
    const q2_sh = [_]c_int{ B, acc_run, Hk, Dk };
    const v2_sh = [_]c_int{ B, acc_run, Hv, Dv };
    const g2_sh = [_]c_int{ B, acc_run, Hv };
    const z3 = [_]c_int{ 1, 1, 1, 1 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    var v2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v2);
    var g2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g2);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_slice(&q2, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &q2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&k2, kk, &[_]c_int{ 0, 0, 0, 0 }, 4, &q2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&v2, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &v2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&g2, g, &[_]c_int{ 0, 0, 0 }, 3, &g2_sh, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_slice(&b2, beta, &[_]c_int{ 0, 0, 0 }, 3, &g2_sh, 3, &[_]c_int{ 1, 1, 1 }, 3, s));

    const ref2 = try gdnTestRun(false, false, q2, k2, v2, g2, b2, st, B, acc_run, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(ref2);

    // Pull position 1 out of the length-T seq run.
    const state_seq_full = try gdnTestRunSeqAt(q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, 1, s);
    defer _ = mlx.mlx_array_free(state_seq_full);

    var r32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r32);
    var p32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p32);
    try mlx.check(mlx.mlx_astype(&r32, ref2, .float32, s));
    try mlx.check(mlx.mlx_astype(&p32, state_seq_full, .float32, s));
    try mlx.check(mlx.mlx_array_eval(r32));
    try mlx.check(mlx.mlx_array_eval(p32));
    const rd = mlx.mlx_array_data_float32(r32) orelse return error.InvalidDtype;
    const pd = mlx.mlx_array_data_float32(p32) orelse return error.InvalidDtype;
    var mid_diff: f32 = 0;
    for (0..sn) |i| mid_diff = @max(mid_diff, @abs(rd[i] - pd[i]));
    try testing.expect(mid_diff < 1e-3);
}

test "GDN vectorized-gate kernel: a per-channel gate that is UNIFORM per head equals the scalar kernel" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    // KDA gates each state element by its own key channel. The invariant that
    // pins the indexing: broadcast a per-head decay across that head's Dk
    // channels and the `_vec` kernel must reproduce the scalar kernel BIT FOR
    // BIT — same recurrence, same reduction order, only the gate fetch differs.
    // A wrong stride/base in the `_vec` arm (reading head 0's channel for every
    // head, or walking Hv instead of Hv*Dk per step) survives every "output is
    // finite and roughly decays" check but fails this one.
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const T: c_int = 5;
    const Hk: c_int = 2;
    const Hv: c_int = 2;
    const Dk: c_int = 128;
    const Dv: c_int = 128;

    var prng = std.Random.DefaultPrng.init(0x11CE);
    const rnd = prng.random();
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const gvn: usize = @intCast(B * T * Hv * Dk);
    const sn: usize = @intCast(B * Hv * Dv * Dk);
    const qd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(qd);
    const kd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(kd);
    const vd = try testing.allocator.alloc(f32, vn);
    defer testing.allocator.free(vd);
    const gd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(gd);
    const gvd = try testing.allocator.alloc(f32, gvn);
    defer testing.allocator.free(gvd);
    const bd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(bd);
    const sd = try testing.allocator.alloc(f32, sn);
    defer testing.allocator.free(sd);
    for (qd) |*x| x.* = rnd.float(f32) - 0.5;
    for (kd) |*x| x.* = rnd.float(f32) - 0.5;
    for (vd) |*x| x.* = rnd.float(f32) - 0.5;
    for (gd) |*x| x.* = 0.5 + 0.4 * rnd.float(f32);
    for (bd) |*x| x.* = rnd.float(f32);
    for (sd) |*x| x.* = rnd.float(f32) - 0.5;
    // Broadcast each (token, head) decay across that head's Dk channels.
    for (0..gn) |i| {
        for (0..@as(usize, @intCast(Dk))) |c| gvd[i * @as(usize, @intCast(Dk)) + c] = gd[i];
    }

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const gvsh = [_]c_int{ B, T, Hv, Dk };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const raw = [_]mlx.mlx_array{
        mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32),
        mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32),
        mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32),
        mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32),
        mlx.mlx_array_new_data(gvd.ptr, &gvsh, 4, .float32),
        mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32),
        mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32),
    };
    defer for (raw) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var bf: [raw.len]mlx.mlx_array = undefined;
    for (raw, 0..) |r, i| {
        bf[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&bf[i], r, .bfloat16, s));
    }
    defer for (bf) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const q = bf[0];
    const kk = bf[1];
    const v = bf[2];
    const g_head = bf[3];
    const g_chan = bf[4];
    const beta = bf[5];
    const st = bf[6];

    const state_scalar = try gdnTestRun(false, false, q, kk, v, g_head, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_scalar);
    const state_vec = try gdnTestRun(false, true, q, kk, v, g_chan, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_vec);

    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, state_scalar, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, state_vec, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd2 = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    var max_diff: f32 = 0;
    var any_nonzero = false;
    for (0..sn) |i| {
        max_diff = @max(max_diff, @abs(ad[i] - bd2[i]));
        if (@abs(ad[i]) > 1e-3) any_nonzero = true;
    }
    // Guard the guard: an all-zero state would pass any diff check.
    try testing.expect(any_nonzero);
    try testing.expectEqual(@as(f32, 0), max_diff);

    // And the seq-capture variant of the same kernel agrees with its own
    // scalar twin, so spec rollback stays correct on a vector-gated arch.
    const seq_scalar = try gdnTestRun(true, false, q, kk, v, g_head, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(seq_scalar);
    const seq_vec = try gdnTestRun(true, true, q, kk, v, g_chan, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(seq_vec);
    var c32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c32);
    var d32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d32);
    try mlx.check(mlx.mlx_astype(&c32, seq_scalar, .float32, s));
    try mlx.check(mlx.mlx_astype(&d32, seq_vec, .float32, s));
    try mlx.check(mlx.mlx_array_eval(c32));
    try mlx.check(mlx.mlx_array_eval(d32));
    const cd = mlx.mlx_array_data_float32(c32) orelse return error.InvalidDtype;
    const dd = mlx.mlx_array_data_float32(d32) orelse return error.InvalidDtype;
    var seq_diff: f32 = 0;
    for (0..sn) |i| seq_diff = @max(seq_diff, @abs(cd[i] - dd[i]));
    try testing.expectEqual(@as(f32, 0), seq_diff);
}

// Like gdnTestRun(seq=true) but returns the state at an arbitrary position `pos`
// (not just the last) — mirrors what `ssmRollbackFromCapture` slices out.
fn gdnTestRunSeqAt(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, pos: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    const ss_shape = [_]c_int{ T, B, Hv, Dv, Dk };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ss_shape, 5, .bfloat16));
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, Dv, B * Hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));
    const seq_stride = mlx.mlx_array_new_int(B * Hv * Dv * Dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar, seq_stride };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try getGdnKernelSeq(false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
    var ss = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ss);
    try mlx.check(mlx.mlx_vector_array_get(&ss, outputs_vec, 1));
    const start = [_]c_int{ pos, 0, 0, 0, 0 };
    const stop = [_]c_int{ pos + 1, B, Hv, Dv, Dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, ss, &start, 5, &stop, 5, &strides, 5, s));
    const fshape = [_]c_int{ B, Hv, Dv, Dk };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, sliced, &fshape, 4, s));
    return out;
}

// ── GDN blocked-seq kernel test harness ──

/// Truncate an f32 to a bf16-representable value (top 16 bits). Test inputs
/// pass through this so the f64 host reference and the bf16 kernels consume
/// IDENTICAL values — the only error left is kernel accumulation/rounding.
pub fn bf16Trunc(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    return @bitCast(bits & 0xFFFF_0000);
}

const GdnHostRef = struct { y: []f32, state: []f32 };

/// f64 host ground truth of the exact GDN recurrence (the stock kernel's
/// math, per house rule: GPU parity is judged vs fp32+ ground truth, never
/// kernel-vs-kernel bf16 agreement).
fn gdnHostRef(
    al: std.mem.Allocator,
    qd: []const f32,
    kd: []const f32,
    vd: []const f32,
    gd: []const f32,
    bd: []const f32,
    sd: []const f32,
    B: usize,
    T: usize,
    Hk: usize,
    Hv: usize,
    Dk: usize,
    Dv: usize,
) !GdnHostRef {
    const y = try al.alloc(f32, B * T * Hv * Dv);
    errdefer al.free(y);
    const st_out = try al.alloc(f32, B * Hv * Dv * Dk);
    errdefer al.free(st_out);
    const S = try al.alloc(f64, Dv * Dk);
    defer al.free(S);
    const group = Hv / Hk;
    for (0..B) |b| {
        for (0..Hv) |hv| {
            const hk = hv / group;
            for (0..Dv) |dvi| {
                for (0..Dk) |dki| S[dvi * Dk + dki] = sd[((b * Hv + hv) * Dv + dvi) * Dk + dki];
            }
            for (0..T) |t| {
                const gt: f64 = gd[(b * T + t) * Hv + hv];
                const bt: f64 = bd[(b * T + t) * Hv + hv];
                const kbase = ((b * T + t) * Hk + hk) * Dk;
                for (S) |*x| x.* *= gt;
                for (0..Dv) |dvi| {
                    var kv: f64 = 0;
                    for (0..Dk) |dki| kv += S[dvi * Dk + dki] * @as(f64, kd[kbase + dki]);
                    const delta = (@as(f64, vd[((b * T + t) * Hv + hv) * Dv + dvi]) - kv) * bt;
                    var out: f64 = 0;
                    for (0..Dk) |dki| {
                        S[dvi * Dk + dki] += @as(f64, kd[kbase + dki]) * delta;
                        out += S[dvi * Dk + dki] * @as(f64, qd[kbase + dki]);
                    }
                    y[((b * T + t) * Hv + hv) * Dv + dvi] = @floatCast(out);
                }
            }
            for (0..Dv) |dvi| {
                for (0..Dk) |dki| st_out[((b * Hv + hv) * Dv + dvi) * Dk + dki] = @floatCast(S[dvi * Dk + dki]);
            }
        }
    }
    return .{ .y = y, .state = st_out };
}

const GdnRunOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// Run the GDN recurrence on one route and return BOTH outputs.
pub fn gdnRunYState(route: GdnRoute, tb_requested: u32, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !GdnRunOut {
    // Mirror the production route exactly: dtypes read off the arrays, block
    // size clamped to what the staging budget allows at that width.
    const in_dtype = mlx.mlx_array_dtype(q);
    const st_dtype = mlx.mlx_array_dtype(state_in);
    const tb = gdnBlockTFor(tb_requested, Dk, mlx.mlx_array_itemsize(q)) orelse return error.GdnBlockedDeclined;
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    // The state leaves at the width it came in (f32 for Hadamard packs), as in the forward.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, st_dtype));
    try gdnSetGrid(config, route, Dv, Hv, B);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", in_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try gdnRouteKernel(route, tb, false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var yo = mlx.mlx_array_new();
    var so = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&yo, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&so, outputs_vec, 1));
    return .{ .y = yo, .state = so };
}

/// `in_dtype` is the ACTIVATION width q/k/v arrive in. bf16 is the common
/// case; an f16 checkpoint promotes its activations to f32 (f16 ⊕ f32 scalar),
/// which is a different kernel specialization AND a different threadgroup
/// budget — both covered by the sweep below.
const GdnCase = struct { B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dv: c_int, tb: u32, in_dtype: mlx.mlx_dtype = .bfloat16, state_dtype: mlx.mlx_dtype = .bfloat16 };

/// Runs one geometry through host ref + stock + blocked and asserts the
/// blocked kernel is no less accurate than the stock kernel it replaces.
fn gdnBlockedParityCase(case: GdnCase, s: mlx.mlx_stream) !void {
    const al = testing.allocator;
    const B = case.B;
    const T = case.T;
    const Hk = case.Hk;
    const Hv = case.Hv;
    const Dk: c_int = 128;
    const Dv = case.Dv;
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);

    var prng = std.Random.DefaultPrng.init(0xB10C5ED);
    const rnd = prng.random();
    const qd = try al.alloc(f32, qn);
    defer al.free(qd);
    const kd = try al.alloc(f32, qn);
    defer al.free(kd);
    const vd = try al.alloc(f32, vn);
    defer al.free(vd);
    const gd = try al.alloc(f32, gn);
    defer al.free(gd);
    const bd = try al.alloc(f32, gn);
    defer al.free(bd);
    const sd = try al.alloc(f32, sn);
    defer al.free(sd);
    for (qd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (kd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (vd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (gd) |*x| x.* = bf16Trunc(0.5 + 0.4 * rnd.float(f32));
    for (bd) |*x| x.* = bf16Trunc(rnd.float(f32));
    for (sd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);

    const ref = try gdnHostRef(al, qd, kd, vd, gd, bd, sd, @intCast(B), @intCast(T), @intCast(Hk), @intCast(Hv), @intCast(Dk), @intCast(Dv));
    defer al.free(ref.y);
    defer al.free(ref.state);

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    // Mirrors the live dtype signature: q/k/v/beta follow the ACTIVATION width
    // (bf16 checkpoints stay bf16; f16 checkpoints promote to f32), `g` is
    // always bf16 because it comes out of our fused gate kernel, and the SSM
    // state buffer is bf16 unless the case asks for f32 (Hadamard packs). The mixed
    // set is the point — the kernel must read each input at its own width.
    try mlx.check(mlx.mlx_astype(&q, q32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&v, v32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&st, st32, case.state_dtype, s));

    const stock = try gdnRunYState(.stock, 16, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(stock.y);
    defer _ = mlx.mlx_array_free(stock.state);
    const stock_y = try evalToF32(al, stock.y, vn, s);
    defer al.free(stock_y);
    const stock_st = try evalToF32(al, stock.state, sn, s);
    defer al.free(stock_st);
    const stock_y_err = maxAbsDiff(stock_y, ref.y);
    const stock_st_err = maxAbsDiff(stock_st, ref.state);

    // No-worse-than-reference: each prefill kernel's error vs the f64 ground
    // truth must not exceed the stock kernel's (all bf16-out; 1.5x headroom
    // covers fp32 summation-order differences, gross bugs blow way past it).
    for ([_]GdnRoute{ .blocked, .pipelined }) |route| {
        const got = try gdnRunYState(route, case.tb, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
        defer _ = mlx.mlx_array_free(got.y);
        defer _ = mlx.mlx_array_free(got.state);
        const got_y = try evalToF32(al, got.y, vn, s);
        defer al.free(got_y);
        const got_st = try evalToF32(al, got.state, sn, s);
        defer al.free(got_st);
        const y_err = maxAbsDiff(got_y, ref.y);
        const st_err = maxAbsDiff(got_st, ref.state);
        if (y_err > 1.5 * stock_y_err + 0.02 or st_err > 1.5 * stock_st_err + 0.02) {
            std.debug.print(
                "GDN {s} parity FAIL (B={d} T={d} Hk={d} Hv={d} Dv={d} tb={d}): y {d:.5} vs stock {d:.5}, state {d:.5} vs stock {d:.5}\n",
                .{ @tagName(route), case.B, case.T, case.Hk, case.Hv, case.Dv, case.tb, y_err, stock_y_err, st_err, stock_st_err },
            );
            return error.GdnBlockedParityFailed;
        }
    }
}

test "GDN pipelined route: measured archs on NAX only" {
    try testing.expect(gdnPipelinedFor("qwen4_exp", true));
    try testing.expect(gdnPipelinedFor("qwen3_5", true));
    try testing.expect(!gdnPipelinedFor("qwen4_exp", false)); // unmeasured chip
    try testing.expect(!gdnPipelinedFor("qwen3_5_moe", true)); // unmeasured arch
    try testing.expect(!gdnPipelinedFor("prism_hadamard_qwen35", true));
}

test "GDN pipelined route: each model type gets its own verdict in one process" {
    // Flash Next first, then an unmeasured MoE pack: the second call must not inherit the first's route.
    const order = [_][]const u8{ "qwen4_exp", "qwen3_5_moe", "prism_hadamard_qwen35", "qwen3_5" };
    for (order) |t| {
        _ = gdnPipelinedEnabled(t);
    }
    for (order) |t| {
        try testing.expectEqual(gdnPipelinedFor(t, naxAvailable()), gdnPipelinedEnabled(t));
    }
}

test "GDN blocked-seq and pipelined kernels: no worse than stock vs f64 ground truth (T/GQA/Dv/TB sweep)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // T deliberately hits non-multiples of every supported TB (16/32/48);
    // Hk<Hv exercises GQA head mapping; Dv 32..128 exercises 1..4 DB blocks.
    const cases = [_]GdnCase{
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32 }, // live-like 27B geometry (fewer heads)
        .{ .B = 2, .T = 64, .Hk = 1, .Hv = 2, .Dv = 32, .tb = 32 }, // batch>1, minimal Dv
        .{ .B = 1, .T = 65, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16 }, // TB=16 variant, ragged tail 1
        .{ .B = 1, .T = 49, .Hk = 2, .Hv = 2, .Dv = 32, .tb = 48 }, // TB=48 variant, ragged tail 1
        // fp32 activations (f16 checkpoints promote to them — the geometry
        // that killed the server on prism-ml/Ternary-Bonsai-27B-mlx-2bit).
        // The kernel must specialize on the real width AND clamp the block
        // size to fit the doubled staging; requesting 32/48 exercises both.
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32, .in_dtype = .float32 },
        .{ .B = 1, .T = 49, .Hk = 2, .Hv = 2, .Dv = 32, .tb = 48, .in_dtype = .float32 },
        .{ .B = 2, .T = 64, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16, .in_dtype = .float32 },
        // f16 activations: same 2-byte staging as bf16, different kernel.
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 64, .tb = 32, .in_dtype = .float16 },
        // f32 GDN state (`ssmStateDtype` for Hadamard packs, Bonsai 2): the kernels'
        // StT = float instantiation, with bf16 and with f32 activations.
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32, .state_dtype = .float32 },
        .{ .B = 1, .T = 65, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16, .in_dtype = .float32, .state_dtype = .float32 },
    };
    for (cases) |case| try gdnBlockedParityCase(case, s);
}

test "GDN blocked-seq kernel: chunk-boundary state continuity (split run == full run)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    try gdnChunkContinuity(.blocked);
}

test "GDN pipelined kernel: chunk-boundary state continuity (split run == full run)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    try gdnChunkContinuity(.pipelined);
}

fn gdnChunkContinuity(route: GdnRoute) !void {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    const B: c_int = 1;
    const T: c_int = 100;
    const T1: c_int = 48; // non-multiple of TB=32: exercises the ragged tail mid-sequence
    const Hk: c_int = 2;
    const Hv: c_int = 4;
    const Dk: c_int = 128;
    const Dv: c_int = 64;
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);

    var prng = std.Random.DefaultPrng.init(0x5EC0DD);
    const rnd = prng.random();
    const qd = try al.alloc(f32, qn);
    defer al.free(qd);
    const kd = try al.alloc(f32, qn);
    defer al.free(kd);
    const vd = try al.alloc(f32, vn);
    defer al.free(vd);
    const gd = try al.alloc(f32, gn);
    defer al.free(gd);
    const bd = try al.alloc(f32, gn);
    defer al.free(bd);
    const sd = try al.alloc(f32, sn);
    defer al.free(sd);
    for (qd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (kd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (vd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (gd) |*x| x.* = bf16Trunc(0.5 + 0.4 * rnd.float(f32));
    for (bd) |*x| x.* = bf16Trunc(rnd.float(f32));
    for (sd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_astype(&q, q32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&v, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&st, st32, .bfloat16, s));

    // Full run.
    const full = try gdnRunYState(route, 32, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(full.y);
    defer _ = mlx.mlx_array_free(full.state);

    // Split run: [0, T1) then [T1, T), carrying the bf16 state — exactly what
    // chunked prefill does (ssm_state hand-off between forwardWith chunks).
    const strides4 = [_]c_int{ 1, 1, 1, 1 };
    const strides3 = [_]c_int{ 1, 1, 1 };
    var q1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q1);
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var v1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v1);
    var g1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g1);
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_slice(&q1, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&k1, kk, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&v1, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hv, Dv }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&g1, g, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T1, Hv }, 3, &strides3, 3, s));
    try mlx.check(mlx.mlx_slice(&b1, beta, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T1, Hv }, 3, &strides3, 3, s));
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    var v2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v2);
    var g2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g2);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_slice(&q2, q, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&k2, kk, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&v2, v, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hv, Dv }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&g2, g, &[_]c_int{ 0, T1, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));
    try mlx.check(mlx.mlx_slice(&b2, beta, &[_]c_int{ 0, T1, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));

    const part1 = try gdnRunYState(route, 32, q1, k1, v1, g1, b1, st, B, T1, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(part1.y);
    defer _ = mlx.mlx_array_free(part1.state);
    const part2 = try gdnRunYState(route, 32, q2, k2, v2, g2, b2, part1.state, B, T - T1, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(part2.y);
    defer _ = mlx.mlx_array_free(part2.state);

    const full_st = try evalToF32(al, full.state, sn, s);
    defer al.free(full_st);
    const split_st = try evalToF32(al, part2.state, sn, s);
    defer al.free(split_st);

    // The only divergence allowed is the one bf16 state rounding injected at
    // the boundary (live chunked prefill injects the identical rounding).
    var max_mag: f32 = 0;
    for (full_st) |x| max_mag = @max(max_mag, @abs(x));
    const tol = 0.02 * max_mag + 0.02;
    const st_diff = maxAbsDiff(full_st, split_st);
    try testing.expect(st_diff < tol);

    // y of the second chunk must also line up with the full run's tail.
    var y_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_tail);
    try mlx.check(mlx.mlx_slice(&y_tail, full.y, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hv, Dv }, 4, &strides4, 4, s));
    const n2: usize = @intCast(B * (T - T1) * Hv * Dv);
    const tail_f = try evalToF32(al, y_tail, n2, s);
    defer al.free(tail_f);
    const part2_y = try evalToF32(al, part2.y, n2, s);
    defer al.free(part2_y);
    try testing.expect(maxAbsDiff(tail_f, part2_y) < tol);
}

test "gdnBlockTFor: the block size follows the INPUT WIDTH, not a bf16 assumption" {
    // bf16/f16 stage 2 bytes — every supported block size fits, so the
    // requested one stands.
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(16, 128, 2));
    try testing.expectEqual(@as(?u32, 32), gdnBlockTFor(32, 128, 2));
    try testing.expectEqual(@as(?u32, 48), gdnBlockTFor(48, 128, 2));

    // fp32 activations (what an f16 checkpoint promotes to) stage 4 bytes:
    // TB=32 needs ~40 KiB and TB=48 ~60 KiB, both past Metal's 32 KiB
    // threadgroup budget, so each clamps DOWN rather than emitting a kernel
    // that fails to compile and takes the server with it.
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(16, 128, 4));
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(32, 128, 4));
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(48, 128, 4));

    // The staging budget the clamp is derived from — pinned so a change to
    // the kernel's threadgroup declarations has to come here too.
    try testing.expect(gdnBlockedTgBytes(48, 128, 2) <= METAL_TG_BYTES);
    try testing.expect(gdnBlockedTgBytes(16, 128, 4) <= METAL_TG_BYTES);
    try testing.expect(gdnBlockedTgBytes(32, 128, 4) > METAL_TG_BYTES);

    // Nothing fits at all → decline the blocked route (null), never fall
    // through to an over-budget dispatch.
    try testing.expectEqual(@as(?u32, null), gdnBlockTFor(48, 128, 16));
    // A request below the smallest supported block declines too.
    try testing.expectEqual(@as(?u32, null), gdnBlockTFor(8, 128, 2));
}

test "gdnBlockedEligible: width floor + exact-128 Dk + Dv/head-group alignment" {
    // Live 27B GDN geometry at prefill width: eligible.
    try testing.expect(gdnBlockedEligible(2048, 128, 128, 16, 32));
    try testing.expect(gdnBlockedEligible(GDN_BLOCKED_MIN_T, 128, 128, 16, 32));
    // Below the width floor (decode, spec verify widths): stock kernel.
    try testing.expect(!gdnBlockedEligible(GDN_BLOCKED_MIN_T - 1, 128, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(1, 128, 128, 16, 32));
    // Off-geometry: Dk != 128, Dv not a multiple of 32, ragged GQA grouping.
    try testing.expect(!gdnBlockedEligible(2048, 96, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 256, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 48, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 16, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 128, 3, 32));
}

test "GDN µbench: sequential kernel vs bare qmm at 27B shapes (attribution; diag gdn-ubench)" {
    // ATTRIBUTION probe, not a pass/fail guard (live A/Bs decide shipping):
    // decomposes the multi-token forward ladder into (a) the GDN recurrence
    // kernel's sequential-over-T cost and (b) qmm row-count effects, at the
    // real Qwen3.6-27B geometry. Run with:
    //   MLX_SERVE_DIAG=gdn-ubench zig build test -Doptimize=ReleaseFast -Dtest-filter="GDN µbench"
    if (env.get(.GDN_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const Hk: c_int = 16;
    const Hv: c_int = 48;
    const Dk: c_int = 128;
    const Dv: c_int = 128;
    const HIDDEN: c_int = 5120;
    const QKVZ_OUT: c_int = 16384; // key_dim*2 + value_dim + z_dim
    const GDN_LAYERS = 48;
    const T_LIST = [_]c_int{ 1, 2, 4, 8, 16, 32, 64, 256, 1024, 2048 };
    const WARM = 3;
    const ITERS = 20;

    var prng = std.Random.DefaultPrng.init(0xBE9C);
    const rnd = prng.random();

    // One shared random pool big enough for the largest T.
    const max_t: usize = 2048;
    const mkbf16 = struct {
        fn f(alloc: std.mem.Allocator, r: std.Random, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try alloc.alloc(f32, n);
            defer alloc.free(buf);
            for (buf) |*x| x.* = r.float(f32) - 0.5;
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, st));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.f;

    const q_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), Hk, Dk }, s);
    defer _ = mlx.mlx_array_free(q_full);
    const k_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), Hk, Dk }, s);
    defer _ = mlx.mlx_array_free(k_full);
    const v_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), Hv, Dv }, s);
    defer _ = mlx.mlx_array_free(v_full);
    const g_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), Hv }, s);
    defer _ = mlx.mlx_array_free(g_full);
    const b_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), Hv }, s);
    defer _ = mlx.mlx_array_free(b_full);
    const st0 = try mkbf16(testing.allocator, rnd, &.{ B, Hv, Dv, Dk }, s);
    defer _ = mlx.mlx_array_free(st0);
    const x_full = try mkbf16(testing.allocator, rnd, &.{ B, @intCast(max_t), HIDDEN }, s);
    defer _ = mlx.mlx_array_free(x_full);

    // 4-bit/gs64 quantized weights at the two big per-layer qmm shapes:
    // the fused qkvz projection [16384, 5120] and an MLP gate [17408, 5120].
    const QTriple = struct { w: mlx.mlx_array, sc: mlx.mlx_array, b: mlx.mlx_array };
    const mkq = struct {
        fn f(alloc: std.mem.Allocator, r: std.Random, rows: c_int, cols: c_int, st: mlx.mlx_stream) !QTriple {
            const shape = [_]c_int{ rows, cols };
            const n: usize = @intCast(rows * cols);
            const buf = try alloc.alloc(f32, n);
            defer alloc.free(buf);
            for (buf) |*x| x.* = r.float(f32) - 0.5;
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var dense = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dense);
            try mlx.check(mlx.mlx_astype(&dense, f32_arr, .bfloat16, st));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            var out: QTriple = .{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
            try mlx.check(mlx.mlx_vector_array_get(&out.w, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&out.sc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&out.b, triple, 2));
            for ([_]mlx.mlx_array{ out.w, out.sc, out.b }) |a| try mlx.check(mlx.mlx_array_eval(a));
            return out;
        }
    }.f;
    const w_qkvz = try mkq(testing.allocator, rnd, QKVZ_OUT, HIDDEN, s);
    defer {
        _ = mlx.mlx_array_free(w_qkvz.w);
        _ = mlx.mlx_array_free(w_qkvz.sc);
        _ = mlx.mlx_array_free(w_qkvz.b);
    }
    const w_mlp = try mkq(testing.allocator, rnd, 17408, HIDDEN, s);
    defer {
        _ = mlx.mlx_array_free(w_mlp.w);
        _ = mlx.mlx_array_free(w_mlp.sc);
        _ = mlx.mlx_array_free(w_mlp.b);
    }

    std.debug.print("\n[gdn-ubench] 27B geometry Hk={d} Hv={d} Dk={d} Dv={d}; {d} timed iters (avg ms per SINGLE dispatch; x{d} = per-model estimate)\n", .{ Hk, Hv, Dk, Dv, ITERS, GDN_LAYERS });
    std.debug.print("[gdn-ubench] {s:>4} {s:>12} {s:>12} {s:>14} {s:>14} {s:>14}\n", .{ "T", "gdn_ms", "gdn_x48_ms", "qkvz_qmm_ms", "mlp_qmm_ms", "qmm_x48x4_ms" });

    for (T_LIST) |T| {
        const strides4 = [_]c_int{ 1, 1, 1, 1 };
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        var beta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_slice(&q, q_full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
        try mlx.check(mlx.mlx_slice(&k, k_full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
        try mlx.check(mlx.mlx_slice(&v, v_full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T, Hv, Dv }, 4, &strides4, 4, s));
        try mlx.check(mlx.mlx_slice(&g, g_full, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));
        try mlx.check(mlx.mlx_slice(&beta, b_full, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));
        try mlx.check(mlx.mlx_slice(&x, x_full, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T, HIDDEN }, 3, &strides3, 3, s));
        for ([_]mlx.mlx_array{ q, k, v, g, beta, x }) |a| try mlx.check(mlx.mlx_array_eval(a));

        // (a) GDN final-state kernel alone.
        var gdn_ms: f64 = 0;
        {
            var it: usize = 0;
            var sw = io_util.Stopwatch.init(tio);
            while (it < WARM + ITERS) : (it += 1) {
                if (it == WARM) sw.reset();
                const out_state = try gdnTestRun(false, false, q, k, v, g, beta, st0, B, T, Hk, Hv, Dk, Dv, s);
                try mlx.check(mlx.mlx_array_eval(out_state));
                _ = mlx.mlx_array_free(out_state);
            }
            gdn_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
        }

        // (b) bare 4-bit qmm at the same T (per-layer weight-read cost).
        const timeQmm = struct {
            fn f(io2: std.Io, xa: mlx.mlx_array, wt: QTriple, st: mlx.mlx_stream) !f64 {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(io2);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_quantized_matmul(&out, xa, wt.w, wt.sc, wt.b, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", st));
                    try mlx.check(mlx.mlx_array_eval(out));
                    _ = mlx.mlx_array_free(out);
                }
                return @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
        }.f;
        const qkvz_ms = try timeQmm(tio, x, w_qkvz, s);
        const mlp_ms = try timeQmm(tio, x, w_mlp, s);

        std.debug.print("[gdn-ubench] {d:>4} {d:>12.3} {d:>12.1} {d:>14.3} {d:>14.3} {d:>14.1}\n", .{
            T,
            gdn_ms,
            gdn_ms * GDN_LAYERS,
            qkvz_ms,
            mlp_ms,
            (qkvz_ms + 3 * mlp_ms) * GDN_LAYERS, // ~4 big qmms per GDN layer (qkvz + gate/up/down-class)
        });
    }
}

test "gdn gate: compiled closure vs graph chain vs prework kernel (bit-identity probe)" {
    const s = mlx.gpuStream();
    const Cb = struct {
        s: mlx.mlx_stream,
        g_dtype: mlx.mlx_dtype,
        fn cb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
            const p: *const @This() = @ptrCast(@alignCast(payload.?));
            var A_log = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
            defer _ = mlx.mlx_array_free(A_log);
            var a = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
            defer _ = mlx.mlx_array_free(a);
            var dt = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&dt, input, 2) != 0) return -1;
            defer _ = mlx.mlx_array_free(dt);
            const g = gdnGateChain(A_log, a, dt, p.g_dtype, p.s) catch return -1;
            const arr = [_]mlx.mlx_array{g};
            res.* = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_array_free(g);
            return 0;
        }
    };

    var prng = std.Random.DefaultPrng.init(0x6A7E);
    const rnd = prng.random();
    const hv: c_int = 48;
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const a_shape = [_]c_int{ 1, 9, hv };
    const a = try attn256RandBf16Scaled(rnd, &a_shape, 16.0, s);
    defer _ = mlx.mlx_array_free(a);

    const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
    const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    // bf16 for a bf16 state, f32 for an f32 one.
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |g_dtype| {
        var payload = Cb{ .s = s, .g_dtype = g_dtype };
        const raw = mlx.mlx_closure_new_func_payload(&Cb.cb, @ptrCast(&payload), null);
        var compiled = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&compiled, raw, true));
        _ = mlx.mlx_closure_free(raw);
        defer _ = mlx.mlx_closure_free(compiled);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        defer _ = mlx.mlx_vector_array_free(out_vec);
        var g_compiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_compiled);
        try mlx.check(mlx.mlx_vector_array_get(&g_compiled, out_vec, 0));
        const g_chain = try gdnGateChain(A_log, a, dt_bias, g_dtype, s);
        defer _ = mlx.mlx_array_free(g_chain);
        try std.testing.expectEqual(g_dtype, mlx.mlx_array_dtype(g_compiled));
        const d = try attn256MaxDiff(g_compiled, g_chain, s);
        std.debug.print("[gate probe] {s}: compiled vs chain max |diff| = {e}\n", .{ @tagName(g_dtype), d });
        // f32: the JIT's transcendentals are not the metallib's, a few ulps apart depending on the GPU; the kernels match the compiled closure.
        try std.testing.expect(d <= if (g_dtype == .float32) 4 * std.math.floatEps(f32) else 0.0);
    }
}

test "gdn gate dtype: an f32 state under bf16 activations takes an f32 gate; Hadamard packs keep theirs" {
    try std.testing.expectEqual(mlx.mlx_dtype.float32, gdn_decode.gateDtype(.bfloat16, .float32));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, gdn_decode.gateDtype(.bfloat16, .bfloat16));
    try std.testing.expectEqual(mlx.mlx_dtype.float16, gdn_decode.gateDtype(.float16, .float32));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, gdnChainGateDtype(.bfloat16, .float32));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, gdnChainGateDtype(.bfloat16, .bfloat16));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, gdnChainGateDtype(.float16, .float32));
}
