//! sdk/blocks/attention/attn256.
//! Owns ATTN256_KERNEL_HEADER, ATTN256_KERNEL_SOURCE, attn256_kernel_cached, getAttn256Kernel, fused256_env_cached, fused256_causal_env_cached, and 29 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const naxSdpaPreferred = @import("sdpa.zig").naxSdpaPreferred;

// ── Fused head_dim-256 prefill attention (flash-style Metal kernel) ──
//
// MLX's fused SDPA (steel_attention) covers head_dim <= 128; every Gemma-4
// and Qwen3.5/3.6 checkpoint ships head_dim 256, which used to fall back to
// the composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
// tensor per layer (26.8 GB/layer at a 102K prompt — the long-context OOM
// class, commit 7550895 budgeted around it). This kernel started as a
// faithful self-contained port of MLX's steel attention (FA-2 online
// softmax, float32 accumulation, exp2 softmax) specialized to BD=256; the
// v2 tiles (BQ=64, BK=32, 8 simdgroups, register-resident Q, uint4 staging
// — see the geometry comment above ATTN256_KERNEL_SOURCE) are our own,
// picked by micro-bench sweep. An optional sliding-window band covers
// Gemma's local layers ("array" mask during prefill) too. The score tensor
// never exists — O(tile) working memory — so the three prefill OOM guards
// (generate.boundedPrefillChunk / prefillEvalCadence / server.prefillMemoryNeeded)
// drop their score term via `prefillHeadDimFused` when this kernel is active.
//
// Scope: seq_len > 1 (prefill), any batch, GQA (Hq % Hk == 0), bf16, causal
// bottom-right alignment (query row r sits at absolute KV position
// kL - qL + r, matching MLX "causal" and createSlidingWindowMask semantics).
// K/V may be non-contiguous cache views (strides used; innermost dim must be
// contiguous — guaranteed for slices along T). Guards and dispatch read one
// shared predicate, so they cannot drift.
pub const ATTN256_KERNEL_HEADER = kernelSource("kernels/attention/attn256_header.metal");

// v2 tile geometry ( micro-bench sweep on the M4 Max, Qwen 24q/4kv
// and Gemma geometries): BQ=64 with 8 simdgroups halves the K/V staging walks
// per query row vs the v1 32-row tile; BK=32 halves the per-column softmax
// rescale + barrier overhead; Q lives in REGISTERS loaded straight from
// global (no Q smem at all — that memory goes to the wider K tile); staging
// runs on uint4 (8 bf16 per instruction, was scalar). Measured vs v1 at
// (qL x kL): (2048x16384) 250 -> 73 ms, (2048x65536) 999 -> 325 ms,
// (8192x8192) 279 -> 77 ms. vs the composed path the v2 kernel wins where
// kL <= ~4*qL (+63% at ratio 1, +15% at 2, +1.8% at 4) and loses beyond
// (-2.3% at 6, -6% at 8) — the causal arm's ratio gate (fusedSdpa256Prefill)
// encodes exactly that envelope. Rejected variants (measured worse): BK=8
// ping-pong (per-block rescale doubles), V-from-global fragments (device
// loads too slow), K-only double-buffer (barrier structure unchanged),
// Q-smem BQ=32 with 2 threadgroups/core (SLC thrash at long kL).
pub const ATTN256_KERNEL_SOURCE = kernelSource("kernels/attention/attn256.metal");

var attn256_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getAttn256Kernel() !mlx.mlx_fast_metal_kernel {
    if (attn256_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" };
    const output_names = [_][*:0]const u8{ "out", "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_p256",
        in_vec,
        out_vec,
        ATTN256_KERNEL_SOURCE,
        ATTN256_KERNEL_HEADER,
        false, // ensure_row_contiguous=false — K/V are cache VIEWS; a forced
        // contiguous copy of the full cache per layer would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    attn256_kernel_cached = kernel;
    return kernel;
}


pub fn fused256Enabled() bool {
    return fused256_override orelse true;
}

/// Plain-CAUSAL dispatch mode. Default `.all` since the kv-chunk dispatch
/// budget landed: the historical net-LOSS of every pre-budget
/// ratio-gated variant (8K: fused-to-ratio-4 231.5, fused-to-ratio-2 232.9,
/// composed 234.7 tok/s — monotonic: more fused = slower, despite the v2
/// kernel winning the µbench at kL <= 2*qL) was the macOS IOGPU
/// interactivity-preemption class: one long dispatch scanning the whole KV
/// gets preempted, and the penalty scales with dispatch length. With the
/// key axis split into budget-sized dispatches (see
/// FUSED256_DEFAULT_DISPATCH_BUDGET) the causal arm WINS the same
/// same-session A/B on the 27B: +2.9%/+2.3%/+4.6% at 8K/16K/32K.
/// The SLIDING-BAND arm (window > 0) has no such competition — composed
/// computes full-width scores + a GB-scale mask while the kernel
/// block-skips outside the band (gemma-26B 99K: 317 -> 712 tok/s) — so it
/// rides the master switch alone and never chunks.
pub const Fused256CausalMode = enum { all, off };

// ── Dispatch work budget (kv-axis chunked dispatch) ──
// One long dispatch scanning the whole KV monopolizes the GPU long enough at
// high kv_len to trip macOS IOGPU interactivity preemption, collapsing
// long-context prefill (oMLX issue #2225: the M3 Max cliff sits near ~1.5e9
// work units ≈ 50 ms/dispatch; design ported from oMLX's
// qwen35_fa256_attention.py, Apache-2.0, oMLX by jundot). Above the budget
// (work = batch·Hq·qL·kL) the CAUSAL arm splits the key axis into
// separately-dispatched BK-aligned chunks with exact online-softmax carry
// (fp32 m/l/O buffers chained between dispatches — bit-identical to the
// single-dispatch result). The band arm never chunks: its in-kernel block
// skip already bounds per-dispatch work by the window.
// The budget mirrors oMLX's 250M fallback (~23 ms/dispatch on the
// M4 Max at the kernel's measured ~1.1e10 work-units/s).
pub const FUSED256_DEFAULT_DISPATCH_BUDGET: i64 = 250_000_000;

/// Test seam: forces the budget without the environment.
pub var fused256_budget_override: ?i64 = null;

pub fn fused256DispatchBudget() i64 {
    return fused256_budget_override orelse FUSED256_DEFAULT_DISPATCH_BUDGET;
}

/// kv-axis chunk length for the budgeted causal dispatch: the largest
/// BK(32)-multiple keeping work-per-dispatch = batch·hq·ql·chunk within
/// budget, floored at one BK block, capped at kl. budget <= 0 = one dispatch.
pub fn fused256KvChunkLen(batch: i64, hq: i64, ql: i64, kl: i64, budget: i64) c_int {
    if (budget <= 0) return @intCast(kl);
    const per_key = batch * hq * ql;
    var chunk = @divTrunc(budget, @max(per_key, 1));
    chunk = @divTrunc(chunk, 32) * 32;
    if (chunk < 32) chunk = 32;
    if (chunk > kl) chunk = kl;
    return @intCast(chunk);
}

/// Test seam: dispatches issued by the LAST fusedSdpa256Prefill call
/// (engagement is counted, never inferred from output equality).
pub var fused256_last_dispatch_count: u32 = 0;

pub fn fused256CausalMode() Fused256CausalMode {
    if (fused256_override) |v| return if (v) .all else .off;
    return if (fused256Enabled()) .all else .off;
}

/// ONE predicate consumed by the three prefill OOM guards AND the dispatch
/// sites: true when prefill attention at this head_dim will NOT materialize
/// the composed [heads, chunk, total_kv] score tensor — either MLX's own
/// fused kernel covers it (<= 128) or our hd-256 kernel does for EVERY arm.
/// Keyed on causal mode: at `.all` (the default since the budgeted-dispatch
/// flip) no hd-256 score tensor materializes; the test seam's composed causal
/// moves the guards' score billing with it. Guards and dispatch must never drift (the effectivePrefillChunk rule).
pub fn prefillHeadDimFused(head_dim: u32) bool {
    return head_dim <= 128 or (head_dim == 256 and fused256CausalMode() == .all);
}

/// Shortest query block the fused hd-256 kernel accepts. Below it MLX's
/// sdpa_vector wins (see the decline in fusedSdpa256Prefill), which also means
/// the kernel is NOT the thing band-masking a spec-verify block —
/// `slidingViewFor` reads the same constant so the trim predicate and the
/// decline can never drift.
pub const FUSED256_MIN_Q_LEN: c_int = 16;

/// Try the fused hd-256 flash prefill kernel (msv_attn_p256). Returns null
/// when a precondition doesn't hold — the caller falls back to the composed
/// path. `window` > 0 adds the sliding-band mask (Gemma local layers,
/// createSlidingWindowMask semantics: key masked when row_abs - col >=
/// window); 0 = plain bottom-right causal. Free function (stream param) so
/// the MTP head can use it too.
///
/// STRUCTURAL precondition (not checkable — lazy arrays carry no strides at
/// graph-build time): q/k/v innermost dim contiguous. True for fresh
/// rope/transpose outputs, cache views (slices along T of a contiguous
/// buffer), and dequantized dense views — i.e. every attention call site.
pub fn fusedSdpa256Prefill(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
) !?mlx.mlx_array {
    // Band arm rides the master switch; plain causal is opt-in (see
    // fused256CausalMode — composed wins live on qwen chunked prefill).
    if (window > 0) {
        if (!fused256Enabled()) return null;
    } else {
        // Opt-in int8 Q/K causal arm on NAX (lossy); null when off or ineligible.
        if (try @import("nax_attention.zig").attention256Int8(s, q, k, v, scale)) |q8| return q8;
        if (fused256CausalMode() == .off) return null;
        // The band arm stays ours (mlx's NAX kernel takes no band); the
        // causal arm yields to the stock fused kernel on NAX.
        if (naxSdpaPreferred()) return null;
    }
    return fusedSdpa256Impl(s, q, k, v, scale, window, null, null);
}

/// The same kernel for the sink-attention arm (gpt_oss / MiMo): any of its head
/// shapes (MiMo's q/k 192, v 128), the layer's learned sinks, causal or banded.
/// MLX has no fused kernel at 192/128, so this one serves on NAX too. Null when
/// it declines (gpt_oss's hd 64 keeps MLX's).
/// No kill switch: MiMo bills no prefill score tensor (`prefillScoreHeadDim` 0),
/// so a composed fallback here would be an unbilled one.
pub fn fusedSinkAttnPrefill(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, window: c_int, sinks: ?mlx.mlx_array) !?mlx.mlx_array {
    // On NAX the tensor-unit kernel runs 192/128; ours takes everything else.
    if (try @import("nax_attention.zig").attention(s, q, k, v, scale, window, sinks)) |out| return out;
    return fusedSdpa256Impl(s, q, k, v, scale, window, null, sinks);
}

/// True when a bf16 view's bound address is 16-byte aligned — the bar every
/// `uint4` staging load in `msv_attn_p256` / `msv_attn_qsa256` needs. A lazy
/// array has no base yet (the kernel binds its buffer at dispatch, like the
/// stride checks in `gatherQsa256` see); an available one returns the address
/// the kernel really reads. Same base test `hcUvEligible` runs for its weights.
pub fn attn256Base16Aligned(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0 or !avail) return true;
    const p = mlx.mlx_array_data_uint32(a) orelse return true;
    return @intFromPtr(p) % 16 == 0;
}

pub var qsa_fused_override: ?bool = null;

/// The test seam's false runs the stock array-mask sdpa (the unfused
/// hd-256 fallback on every silicon: `use_fallback` has no fused arm for an
/// array mask at hd 256).
pub fn qsaFusedEnabled() bool {
    return qsa_fused_override orelse true;
}

var qsa_fused_logged = false;

/// Array-mask arm of msv_attn_p256 (qwen4_exp QSA prefill): `mask` is the
/// [B,1,qL,kL] bool visibility with causal folded in. Key blocks no row of a
/// 64-row q tile sees are skipped in-kernel via a per-tile any-visible table
/// built here. Same envelope as the causal arm (q >= 16, hd 256, bf16).
pub fn fusedSdpa256Masked(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!fused256Enabled() or !qsaFusedEnabled()) return null;
    if (mask.ctx == null or mlx.mlx_array_ndim(mask) != 4 or mlx.mlx_array_dtype(mask) != .bool_) return null;
    const ms = mlx.getShape(mask);
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (ms[0] != qs[0] or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const out = try fusedSdpa256Impl(s, q, k, v, scale, 0, mask, null);
    if (out != null and !qsa_fused_logged) {
        qsa_fused_logged = true;
        log.info("[qsa-fused] engaged: msv_attn_p256 mask arm qL={d} kL={d} Hq={d} Hkv={d}\n", .{ qs[2], ks[2], qs[1], ks[1] });
    }
    return out;
}

/// Per-(batch, q tile, key block) any-visible table for the QSA arm:
/// [B, ceil(qL/64), ceil(kL/32)] bool.
fn qsaSkipTable(s: mlx.mlx_stream, mask: mlx.mlx_array) !mlx.mlx_array {
    const ms = mlx.getShape(mask);
    const nq: c_int = @divTrunc(ms[2] + 63, 64);
    const nk: c_int = @divTrunc(ms[3] + 31, 32);
    const axes = [_]c_int{ 2, 3 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ nq * 64 - ms[2], nk * 32 - ms[3] };
    const falsev = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(falsev);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, mask, &axes, 2, &low, 2, &high, 2, falsev, "constant", s));
    var r5 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r5);
    try mlx.check(mlx.mlx_reshape(&r5, padded, &[_]c_int{ ms[0], nq, 64, nk, 32 }, 5, s));
    const red = [_]c_int{ 2, 4 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_any_axes(&out, r5, &red, 2, false, s));
    return out;
}

pub fn fusedSdpa256Impl(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    /// Learned per-head softmax sinks [Hq] (gpt_oss / MiMo), or null.
    sinks: ?mlx.mlx_array,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    // q/k 256 with v 256, or MiMo's q/k 192 with v 128.
    if (qs[3] != ks[3] or !((qs[3] == 256 and vs[3] == 256) or (qs[3] == 192 and vs[3] == 128))) return null;
    const bdv: c_int = vs[3];
    // Short sequences (< 16: decode AND spec-decode VERIFY forwards) belong
    // to MLX's sdpa_vector, which covers hd 256 natively and beats a 64-row
    // prefill tile walking the whole KV for a few-row query — dispatching
    // those here measured decode 48 -> 18 tok/s at 4K ctx on the 27B (MTP
    // verify is seq 1+depth, up to 9 at the NAX depth-8 cap — with causal
    // fused default-on the floor must clear it). 16 matches oMLX's
    // _MIN_ROUTE_Q_LEN; genuine prefill chunks are always far above it.
    if (qs[2] < FUSED256_MIN_Q_LEN) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    const attn_dt = mlx.mlx_array_dtype(q);
    if ((attn_dt != .bfloat16 and attn_dt != .float16) or mlx.mlx_array_dtype(k) != attn_dt or mlx.mlx_array_dtype(v) != attn_dt) return null;

    // K/V stage through `uint4` loads: an unaligned materialized view base
    // (e.g. the [3,259) slice of a padded cache) reads wrong values; decline
    // and the caller's fallback runs the same shape.
    if (!attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;

    const kernel = getAttn256Kernel() catch return null;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const win_data = [_]i32{@intCast(window)};
    const win = mlx.mlx_array_new_data(&win_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(win);

    // Budgeted kv-axis chunking (CAUSAL arm only — the band arm's in-kernel
    // block skip already bounds per-dispatch work by the window, so chunking
    // it would only add carry traffic). Chunk boundaries are BK(32)-aligned;
    // the fp32 m/l/O carry chained between dispatches keeps the result
    // bit-identical to a single dispatch (see the budget comment above).
    const kL: c_int = ks[2];
    const chunk_len: c_int = if (window > 0)
        kL
    else
        fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget());

    // Dummy 1-elem f32 stands in for absent carry inputs / unwritten outputs.
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    // 4-D so `mask_strides[..3]` exists for the QSA=0 template too.
    const dummy_b_data = [_]bool{false};
    const dummy_b_shape = [_]c_int{ 1, 1, 1, 1 };
    const dummy_b = mlx.mlx_array_new_data(&dummy_b_data, &dummy_b_shape, 4, .bool_);
    defer _ = mlx.mlx_array_free(dummy_b);
    var skip = mlx.mlx_array{ .ctx = null };
    defer if (skip.ctx != null) {
        _ = mlx.mlx_array_free(skip);
    };
    if (mask) |m| skip = try qsaSkipTable(s, m);

    var m_prev = mlx.mlx_array{ .ctx = null };
    var l_prev = mlx.mlx_array{ .ctx = null };
    var o_prev = mlx.mlx_array{ .ctx = null };
    defer {
        if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
        if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
        if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    var dispatches: u32 = 0;
    var k0: c_int = 0;
    while (k0 < kL) : (k0 += chunk_len) {
        const k1: c_int = @min(k0 + chunk_len, kL);
        const final = k1 == kL;
        const has_carry = k0 > 0;

        const two = [_]c_int{2};
        const kr_data = [_]i32{ @intCast(k0), @intCast(k1) };
        const kr = mlx.mlx_array_new_data(&kr_data, &two, 1, .int32);
        defer _ = mlx.mlx_array_free(kr);
        const phase_data = [_]i32{(if (has_carry) @as(i32, 1) else 0) | (if (final) @as(i32, 2) else 0)};
        const phase = mlx.mlx_array_new_data(&phase_data, &one, 1, .int32);
        defer _ = mlx.mlx_array_free(phase);

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        if (final) {
            const o_shape = [_]c_int{ qs[0], qs[1], qs[2], bdv };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, attn_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
            const oc_shape = [_]c_int{ qs[0], qs[1], qs[2], bdv };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, attn_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oc_shape, 4, .float32));
        }
        // One threadgroup (32,8,1) per 64-row q tile per head per batch; grid
        // is in THREADS (dispatch_threads), padded to whole tiles so every
        // threadgroup is full (the cooperative staging loops assume NT=256).
        const nq_tiles: c_int = @divTrunc(qs[2] + 63, 64);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 32, qs[1] * 8, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", attn_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QSA", if (mask != null) 1 else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDV", bdv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SINK", if (sinks != null) 1 else 0));

        const inputs_arr = [_]mlx.mlx_array{
            q,                                k,
            v,                                scl,
            win,                              kr,
            phase,                            if (has_carry) m_prev else dummy,
            if (has_carry) l_prev else dummy, if (has_carry) o_prev else dummy,
            if (mask) |m| m else dummy_b,     if (mask != null) skip else dummy_b,
            sinks orelse dummy,
        };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 4) return error.MetalKernelBadOutputCount;
        dispatches += 1;

        if (final) {
            try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
        } else {
            var m_new = mlx.mlx_array_new();
            var l_new = mlx.mlx_array_new();
            var o_new = mlx.mlx_array_new();
            errdefer {
                if (m_new.ctx != null) _ = mlx.mlx_array_free(m_new);
                if (l_new.ctx != null) _ = mlx.mlx_array_free(l_new);
                if (o_new.ctx != null) _ = mlx.mlx_array_free(o_new);
            }
            try mlx.check(mlx.mlx_vector_array_get(&m_new, outputs_vec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&l_new, outputs_vec, 2));
            try mlx.check(mlx.mlx_vector_array_get(&o_new, outputs_vec, 3));
            if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
            if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
            if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
            m_prev = m_new;
            l_prev = l_new;
            o_prev = o_new;
            m_new = .{ .ctx = null };
            l_new = .{ .ctx = null };
            o_new = .{ .ctx = null };
        }
    }
    fused256_last_dispatch_count = dispatches;
    return out;
}

pub fn attn256MaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // Raw data-pointer reads below: force row-major first. The fused sdpa
    // kernels hand back a [B,L,H,D]-strided VIEW and astype keeps the strides.
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    return max_diff;
}

/// Test seam: forces BOTH hd-256 kernel arms on/off.
pub var fused256_override: ?bool = null;

// ── tests ──

/// Composed reference for the fused kernel tests: MLX's own SDPA (which takes
/// the composed matmul→softmax→matmul path at head_dim 256).
pub fn attn256Reference(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mode: [*:0]const u8,
    mask: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var ref = mlx.mlx_array_new();
    const none = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, mode, if (mask.ctx != null) mask else none, .{ .ctx = null }, false, s));
    return ref;
}

pub fn attn256RandBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| x.* = rnd.float(f32) - 0.5;
    const f32arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f32arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32arr, .bfloat16, s));
    return out;
}

pub fn attn256RandBf16Scaled(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const unit = try attn256RandBf16(rnd, shape, s);
    defer _ = mlx.mlx_array_free(unit);
    const k = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(k);
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_multiply(&wide, unit, k, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, wide, .bfloat16, s));
    return out;
}

/// Largest |element| of an array: the scale a bf16 rounding bar is relative to.
fn attn256MaxAbs(a: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    var m: f32 = 0;
    for (0..n) |i| m = @max(m, @abs(ad[i]));
    return m;
}

/// Two bf16 ulps at the reference's own scale.
pub fn attn256UlpBar(ref: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const m = try attn256MaxAbs(ref, s);
    return 2.0 * m * 0.00390625 + 1.0e-6;
}

/// Cosine of two attention outputs. A max-abs bar alone cannot see a uniform
/// scale error (a dropped or double-counted key shifts every row a little),
/// so the gather parity tests assert BOTH.
pub fn attn256Cosine(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        dot += @as(f64, ad[i]) * @as(f64, bd[i]);
        na += @as(f64, ad[i]) * @as(f64, ad[i]);
        nb += @as(f64, bd[i]) * @as(f64, bd[i]);
    }
    if (na == 0 or nb == 0) return error.ZeroNorm;
    return @floatCast(dot / (@sqrt(na) * @sqrt(nb)));
}

/// Materialized `[B,H,T,264]` padded cache sliced on the last axis to
/// `[lo, lo+256)`: a 256-wide view whose row stride is divisible by 8 elements
/// (16 bytes) while its base sits `lo*2` bytes into the buffer — the shape the
/// `uint4` staging loads of the two old kernels must not read.
pub fn attn256SliceLastDim(
    parent: mlx.mlx_array,
    lo: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const ps = mlx.getShape(parent);
    const one = [_]c_int{ 1, 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const stop = [_]c_int{ ps[0], ps[1], ps[2], lo + 256 };
    try mlx.check(mlx.mlx_slice(&out, parent, &[_]c_int{ 0, 0, 0, lo }, 4, &stop, 4, &one, 4, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}
