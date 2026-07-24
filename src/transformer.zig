const std = @import("std");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const kv_quant = @import("kv_quant.zig");

pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const KVQuantScheme = kv_quant.Scheme;

/// `std.meta.fields` was replaced by parallel `fieldNames`/`fieldTypes`
/// arrays — zip them into the `.name`/`.type` shape every call site below
/// already expects.
const WeightField = struct { name: [:0]const u8, type: type };

fn structFields(comptime T: type) []const WeightField {
    return comptime blk: {
        const names = std.meta.fieldNames(T);
        const types = std.meta.fieldTypes(T);
        var result: [names.len]WeightField = undefined;
        for (names, types, 0..) |name, ty, i| result[i] = .{ .name = name, .type = ty };
        const final = result;
        break :blk &final;
    };
}

// ── GatedDeltaNet fused Metal kernel ──
// Ported from mlx-lm/models/gated_delta.py: `_make_gated_delta_kernel(has_mask=False, vectorized=False)`.
// Processes the entire T-step delta recurrence in a single kernel dispatch, eliminating
// the per-token kernel-launch overhead that otherwise caps prefill at ~330 tok/s on
// Qwen 3.5/3.6 MoE. Template args (Dk, Dv, Hk, Hv) specialize the kernel; inputs carry
// the runtime shapes. State math runs in float32 for numerical stability regardless
// of the input/state storage dtype.
const GDN_KERNEL_SOURCE =
    \\auto n = thread_position_in_grid.z;
    \\auto b_idx = n / Hv;
    \\auto hv_idx = n % Hv;
    \\auto hk_idx = hv_idx / (Hv / Hk);
    \\constexpr int n_per_t = Dk / 32;
    \\
    \\auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\
    \\auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    \\y += b_idx * T * Hv * Dv + hv_idx * Dv;
    \\
    \\auto dk_idx = thread_position_in_threadgroup.x;
    \\auto dv_idx = thread_position_in_grid.y;
    \\
    \\auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    \\auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    \\
    \\float state[n_per_t];
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  state[i] = static_cast<float>(i_state[s_idx]);
    \\}
    \\
    \\auto g_ = g + b_idx * T * Hv;
    \\auto beta_ = beta + b_idx * T * Hv;
    \\
    \\for (int t = 0; t < T; ++t) {
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] * g_[hv_idx];
    \\    kv_mem += state[i] * k_[s_idx];
    \\  }
    \\  kv_mem = simd_sum(kv_mem);
    \\
    \\  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    \\
    \\  float out = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] + k_[s_idx] * delta;
    \\    out += state[i] * q_[s_idx];
    \\  }
    \\  out = simd_sum(out);
    \\  if (thread_index_in_simdgroup == 0) {
    \\    y[dv_idx] = static_cast<InT>(out);
    \\  }
    \\  q_ += Hk * Dk;
    \\  k_ += Hk * Dk;
    \\  v_ += Hv * Dv;
    \\  y += Hv * Dv;
    \\  g_ += Hv;
    \\  beta_ += Hv;
    \\}
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  o_state[s_idx] = static_cast<StT>(state[i]);
    \\}
;

var gdn_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getGdnKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_kernel_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = [_][*:0]const u8{ "y", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "gated_delta_step",
        in_vec,
        out_vec,
        GDN_KERNEL_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_kernel_cached = kernel;
    return kernel;
}

// Per-position-state variant of the GDN recurrence kernel, used ONLY on the
// spec verify forward (small T). Identical math to GDN_KERNEL_SOURCE, but in
// addition to the FINAL state (`state_out`, written from registers like the
// stock kernel) it records the state after every INTERMEDIATE timestep into
// `state_seq` ([T, B, Hv, Dv, Dk]). That lets partial-accept rollback restore
// the accepted-position SSM state by slicing — no re-forward of the accepted
// prefix (the costly part on a 48-layer GatedDeltaNet trunk). The decode and
// prefill paths keep using the original single-state kernel, so this adds zero
// cost outside speculative decoding. `seq_stride` = (B*Hv)*Dv*Dk = per-timestep
// element stride into `state_seq` (passed as a scalar, mirroring `T`).
//
// Capture-tail trim: state_seq[T-1] is deliberately NEVER written — a partial
// accept reads index `accepted` <= T-2 (index T-1 would be a full accept,
// which keeps `state_out` via the normal flow), so the last row was a
// redundant global write + forced the engine's final state to be a slice VIEW
// pinning the whole [T,...] capture buffer across rounds.
const GDN_KERNEL_SEQ_SOURCE =
    \\auto n = thread_position_in_grid.z;
    \\auto b_idx = n / Hv;
    \\auto hv_idx = n % Hv;
    \\auto hk_idx = hv_idx / (Hv / Hk);
    \\constexpr int n_per_t = Dk / 32;
    \\
    \\auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    \\
    \\auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    \\y += b_idx * T * Hv * Dv + hv_idx * Dv;
    \\
    \\auto dk_idx = thread_position_in_threadgroup.x;
    \\auto dv_idx = thread_position_in_grid.y;
    \\
    \\auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    \\auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    \\auto seq_base = (n * Dv + dv_idx) * Dk;
    \\
    \\float state[n_per_t];
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  state[i] = static_cast<float>(i_state[s_idx]);
    \\}
    \\
    \\auto g_ = g + b_idx * T * Hv;
    \\auto beta_ = beta + b_idx * T * Hv;
    \\
    \\for (int t = 0; t < T; ++t) {
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] * g_[hv_idx];
    \\    kv_mem += state[i] * k_[s_idx];
    \\  }
    \\  kv_mem = simd_sum(kv_mem);
    \\
    \\  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
    \\
    \\  float out = 0.0f;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    state[i] = state[i] + k_[s_idx] * delta;
    \\    out += state[i] * q_[s_idx];
    \\  }
    \\  out = simd_sum(out);
    \\  if (thread_index_in_simdgroup == 0) {
    \\    y[dv_idx] = static_cast<InT>(out);
    \\  }
    \\  if (t + 1 < T) {
    \\    auto t_state = state_seq + t * seq_stride + seq_base;
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      t_state[s_idx] = static_cast<StT>(state[i]);
    \\    }
    \\  }
    \\  q_ += Hk * Dk;
    \\  k_ += Hk * Dk;
    \\  v_ += Hv * Dv;
    \\  y += Hv * Dv;
    \\  g_ += Hv;
    \\  beta_ += Hv;
    \\}
    \\for (int i = 0; i < n_per_t; ++i) {
    \\  auto s_idx = n_per_t * dk_idx + i;
    \\  o_state[s_idx] = static_cast<StT>(state[i]);
    \\}
;

var gdn_kernel_seq_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getGdnKernelSeq() !mlx.mlx_fast_metal_kernel {
    if (gdn_kernel_seq_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" };
    const output_names = [_][*:0]const u8{ "y", "state_seq", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "gated_delta_step_seq",
        in_vec,
        out_vec,
        GDN_KERNEL_SEQ_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_kernel_seq_cached = kernel;
    return kernel;
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
// Kill switch: MLX_SERVE_GDN_BLOCKED=0; block size: MLX_SERVE_GDN_BLOCK_T
// (16|32|48, default 32 for bf16 — Metal's 32 KiB threadgroup limit
// governs; fp32 inputs would need 16, but our GDN inputs are always bf16).

/// Test seam: forces the blocked-prefill route on/off without the environment.
pub var gdn_blocked_override: ?bool = null;
var gdn_blocked_env_cached: ?bool = null;

pub fn gdnBlockedEnabled() bool {
    if (gdn_blocked_override) |v| return v;
    if (gdn_blocked_env_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_GDN_BLOCKED");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    gdn_blocked_env_cached = enabled;
    return enabled;
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

const GDN_BLOCKED_TBS = [_]u32{ 16, 32, 48 };
var gdn_blocked_cached: [GDN_BLOCKED_TBS.len]?mlx.mlx_fast_metal_kernel = @splat(null);
var gdn_block_t_cached: ?u32 = null;

pub fn gdnBlockT() u32 {
    if (gdn_block_t_cached) |v| return v;
    const v: u32 = blk: {
        const raw = std.c.getenv("MLX_SERVE_GDN_BLOCK_T") orelse break :blk 32;
        const parsed = std.fmt.parseInt(u32, std.mem.sliceTo(raw, 0), 10) catch break :blk 32;
        for (GDN_BLOCKED_TBS) |cand| {
            if (cand == parsed) break :blk parsed;
        }
        break :blk 32;
    };
    gdn_block_t_cached = v;
    return v;
}

// Body of the blocked-seq kernel; the `constexpr int TB = N;` line is
// prepended per block size by gdnBlockedSource (one cached kernel per TB,
// DISTINCT names — two specializations sharing a name bind the wrong binary).
// Faithful transcription of oMLX's _KERNEL_S_SRC with one generalization:
// state loads/stores go through vec<StT,4> (theirs hard-codes float4 — their
// state is fp32, ours rides the bf16 ssm_state buffer exactly like the stock
// kernel, so cross-chunk behavior is unchanged).
const GDN_KERNEL_BLOCKED_BODY =
    \\constexpr int DB = 32;                             // dv rows per threadgroup
    \\const int tid = thread_position_in_threadgroup.x;  // 0..255
    \\const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    \\const int hv  = threadgroup_position_in_grid.y;
    \\const int b   = threadgroup_position_in_grid.z;
    \\const int hk  = hv / (Hv / Hk);
    \\const int dv0 = blk * DB;
    \\
    \\// thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    \\// the same simdgroup (lane = (dvr%4)*8 + seg).
    \\const int dvr = tid / 8;            // 0..31
    \\const int seg = tid % 8;            // 0..7
    \\const int d0  = seg * 16;
    \\
    \\threadgroup InT k_s[TB][Dk + 8];
    \\threadgroup InT q_s[TB][Dk + 8];
    \\threadgroup InT v_s[TB][DB + 8];
    \\threadgroup float g_s[TB];
    \\threadgroup float b_s[TB];
    \\
    \\const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    \\const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    \\const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    \\const size_t krow = (size_t)Hk * Dk;
    \\
    \\// state fragment in registers: [dv0+dvr][d0..d0+16]
    \\float4 st[4];
    \\{
    \\    const device vec<StT,4>* S_in = (const device vec<StT,4>*)(
    \\        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    \\    for (int i = 0; i < 4; ++i) st[i] = float4(S_in[i]);
    \\}
    \\
    \\device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;
    \\
    \\for (int t0 = 0; t0 < T; t0 += TB) {
    \\    const int tt = min(TB, T - t0);
    \\    // cooperative staging (coalesced): k/q rows, v slice, g/beta
    \\    for (int p = tid; p < tt * Dk; p += 256) {
    \\        const int r = p / Dk, d = p % Dk;
    \\        k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
    \\        q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
    \\    }
    \\    for (int p = tid; p < tt * DB; p += 256) {
    \\        const int r = p / DB, d = p % DB;
    \\        v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
    \\    }
    \\    for (int p = tid; p < tt; p += 256) {
    \\        g_s[p] = (float)g[((size_t)b * T + t0 + p) * Hv + hv];
    \\        b_s[p] = (float)beta[((size_t)b * T + t0 + p) * Hv + hv];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (int t = 0; t < tt; ++t) {
    \\        const float gt = g_s[t];
    \\        const float bt = b_s[t];
    \\        const threadgroup vec<InT,4>* k4 =
    \\            (const threadgroup vec<InT,4>*)&k_s[t][d0];
    \\        const threadgroup vec<InT,4>* q4 =
    \\            (const threadgroup vec<InT,4>*)&q_s[t][d0];
    \\        float4 kf[4];
    \\        for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
    \\        // kv_mem = (g*state) . k ; decay applied to state first
    \\        float4 p4 = 0.0f;
    \\        for (int i = 0; i < 4; ++i) {
    \\            st[i] *= gt;
    \\            p4 += st[i] * kf[i];
    \\        }
    \\        float part = p4.x + p4.y + p4.z + p4.w;
    \\        // reduce across the 8 segment-threads of this dv row
    \\        part += simd_shuffle_down(part, 4);
    \\        part += simd_shuffle_down(part, 2);
    \\        part += simd_shuffle_down(part, 1);
    \\        const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
    \\        const float delta = ((float)v_s[t][dvr] - kv_mem) * bt;
    \\
    \\        float4 o4 = 0.0f;
    \\        for (int i = 0; i < 4; ++i) {
    \\            st[i] += kf[i] * delta;
    \\            o4 += st[i] * float4(q4[i]);
    \\        }
    \\        float out = o4.x + o4.y + o4.z + o4.w;
    \\        out += simd_shuffle_down(out, 4);
    \\        out += simd_shuffle_down(out, 2);
    \\        out += simd_shuffle_down(out, 1);
    \\        if (seg == 0) {
    \\            y_base[(size_t)(t0 + t) * Hv * Dv + dvr] = (InT)out;
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\{
    \\    device vec<StT,4>* S_out = (device vec<StT,4>*)(
    \\        state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    \\    for (int i = 0; i < 4; ++i) S_out[i] = vec<StT,4>(st[i]);
    \\}
;

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

// ── Verify-width split-K quantized matmul (spec-decode fast path) ──
//
// Stock MLX qmm is tuned for M=1 decode (qmv) and large-M prefill (steel);
// the M=2..8-row shapes of speculative VERIFY forwards (and small decode
// batches) fall in a dead zone that underuses memory bandwidth, and at huge
// N (lm_head) the tiny-tile grid thrashes the scheduler. This kernel is a
// port of MTPLX's split-K verify family (mtplx/verify_kernels.py,
// Apache-2.0; design + measurement ledger in that file's header — their
// in-context winner over the wide multi-simdgroup tile): each threadgroup
// owns FOUR output columns, the K reduction splits across K_PARTS
// simdgroups (2 for big N, 4 for small N — deep occupancy queues + latency
// hiding under mixed scheduling with attention/GDN kernels), per-column
// sequential dequant+FMA chains (the interleaved construction measurably
// loses), fp32 accumulation, one threadgroup-barrier partial reduction.
// MROWS/GS/K_PARTS ride as template ints so MLX caches one specialization
// per shape class. 4-bit affine only (the shipping trunk class); everything
// else falls through to stock qmm. Numerics: fp32 accumulate in a different
// order than stock → bf16 tail-ULP class differences (same accepted class
// as every fused kernel here); parity pinned by the verifyQmm test.
// Kill switch: MLX_SERVE_VERIFY_QMM=0.
//
// NAX lane (M5-class machines): on "applegpu_g17" GPUs under macOS >= 26.2
// a fixed 16x32x16 tensor-ops tile (MetalPerformancePrimitives matmul2d on
// the per-core matrix units) extends the family to M 8..16 — past the
// plain-SIMD register cliff at M=8 (T(7) rounds measured 636 ms vs 115
// stock on the M4) — covering deep MTP verify widths (seq 8..9), the merged
// MTP-head forward at high accepted counts, lm_head at M 8..16, and small
// decode batches. Third-generation port; keep the attribution chain:
// DFlash (arXiv:2602.06036) → bstnxbt/dflash-mlx verify_qmm.py (Apache-2.0)
// → MTPLX nax_verify.py (Apache-2.0: the M-padding dispatch, env gating,
// availability probes) → us (plan: todo-m5-nax.md). The kernel object is
// NEVER built — not just never dispatched — where the probe is false:
// matmul2d<.., execution_simdgroup> pipeline creation can fail on non-G17
// hardware. Switches: MLX_SERVE_VERIFY_QMM_NAX=0 kills the lane;
// MLX_SERVE_FORCE_GPU_FAMILY_
// FALLBACK=1 pretends the units are absent (QA rehearsal of the exact
// M1-M4 path on an M5); MLX_SERVE_VERIFY_QMM_NAX_MIN_M lowers the NAX
// takeover width (default 8) for the M5-day A/B of routing M 5..7 to NAX
// (their dispatcher keeps plain SIMD through M=6; our SIMD lanes differ —
// measure, don't inherit).
/// Comptime codegen of the per-M split-K kernel body. The row loads, weight
/// loads, and dequant+FMA chains are emitted as NAMED SCALARS with LITERAL
/// accumulator indices — the ledger's load-bearing constraint: array-indexed
/// rows behind a runtime-ish loop stack-spill (measured 10x at M=6 with the
/// `Vec8 v[MROWS]` form of this same kernel). GS and K_PARTS stay template
/// ints; M is baked into the source (one cached kernel per M, distinct Metal
/// host_names — two specializations sharing a name bind the wrong binary).
fn verifyQmmSource(comptime m: usize, comptime bn: usize) [:0]const u8 {
    comptime {
        const nacc = bn * m;
        var body: []const u8 = "";
        // Row activation loads, all up front.
        for (0..m) |r| {
            body = body ++ std.fmt.comptimePrint("  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r });
        }
        // Weight-word loads for the owned columns, then scales/biases.
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  uint32_t p{d} = w_q[(n0 + {d}) * K_by_p + pack];\n", .{ j, j });
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j });
        }
        // One sequential dequant+FMA chain per output column (column-major
        // construction; the interleaved form measurably loses).
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  {{\n    uint32_t packed = p{d}; float sj = s{d}; float bj = b{d};\n    for (int ki = 0; ki < 8; ++ki) {{\n      float wv = float((packed >> (ki * 4)) & 0xFu) * sj + bj;\n", .{ j, j, j });
            for (0..m) |r| {
                body = body ++ std.fmt.comptimePrint("      acc[{d}] += float(v{d}[ki]) * wv;\n", .{ j * m + r, r });
            }
            body = body ++ "    }\n  }\n";
        }
        var acc_init: []const u8 = "";
        for (0..nacc) |i| {
            acc_init = acc_init ++ std.fmt.comptimePrint("acc[{d}] = 0.0f; ", .{i});
        }
        var acc_sum: []const u8 = "";
        for (0..nacc) |i| {
            acc_sum = acc_sum ++ std.fmt.comptimePrint("acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i });
        }
        return std.fmt.comptimePrint(
            \\auto part = simdgroup_index_in_threadgroup;
            \\auto lane = thread_index_in_simdgroup;
            \\auto tg_n = threadgroup_position_in_grid.y;
            \\
            \\int K = int(K_size);
            \\int N = int(N_size);
            \\int K_by_p = K / 8;
            \\int K_by_gs = K / GS;
            \\int per_part = K_by_p / K_PARTS;
            \\int n0 = int(tg_n) * {d};
            \\int p_start = int(part) * per_part;
            \\int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
            \\
            \\float acc[{d}];
            \\{s}
            \\
            \\using Vec8 = vec<T, 8>;
            \\const device Vec8 *xv = (const device Vec8*)x;
            \\
            \\for (int pack = p_start + int(lane); pack < p_end; pack += 32) {{
            \\  int k_base = pack * 8;
            \\  int gi = k_base / GS;
            \\{s}
            \\}}
            \\
            \\{s}
            \\
            \\threadgroup float partials[K_PARTS * {d}];
            \\if (lane == 0) {{
            \\  _Pragma("unroll")
            \\  for (int i = 0; i < {d}; ++i) {{
            \\    partials[int(part) * {d} + i] = acc[i];
            \\  }}
            \\}}
            \\threadgroup_barrier(mem_flags::mem_threadgroup);
            \\
            \\if (part == 0 && lane < {d}) {{
            \\  float total = 0.0f;
            \\  _Pragma("unroll")
            \\  for (int p = 0; p < K_PARTS; ++p) {{
            \\    total += partials[p * {d} + int(lane)];
            \\  }}
            \\  int j = int(lane) / {d};
            \\  int row = int(lane) - j * {d};
            \\  y[row * N + n0 + j] = T(total);
            \\}}
        , .{ bn, nacc, acc_init, body, acc_sum, nacc, nacc, nacc, nacc, nacc, m, m });
    }
}

/// Wide multi-simdgroup tile for HUGE N (the lm_head class): NSG independent
/// simdgroups ride one threadgroup purely to give the scheduler fewer,
/// heavier units — the split-K tiny-tile grid measurably thrashes there
/// (2.1x stock at M=4), while this tile holds 1.18x off the weight-stream
/// floor per the source ledger. Full-K per simdgroup, no barrier.
fn verifyQmmMsgSource(comptime m: usize, comptime bn: usize) [:0]const u8 {
    comptime {
        const nacc = bn * m;
        var body: []const u8 = "";
        for (0..m) |r| {
            body = body ++ std.fmt.comptimePrint("  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r });
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  uint32_t p{d} = w_q[(n0 + {d}) * K_by_p + pack];\n", .{ j, j });
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j });
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  {{\n    uint32_t packed = p{d}; float sj = s{d}; float bj = b{d};\n    for (int ki = 0; ki < 8; ++ki) {{\n      float wv = float((packed >> (ki * 4)) & 0xFu) * sj + bj;\n", .{ j, j, j });
            for (0..m) |r| {
                body = body ++ std.fmt.comptimePrint("      acc[{d}] += float(v{d}[ki]) * wv;\n", .{ j * m + r, r });
            }
            body = body ++ "    }\n  }\n";
        }
        var acc_init: []const u8 = "";
        for (0..nacc) |i| {
            acc_init = acc_init ++ std.fmt.comptimePrint("acc[{d}] = 0.0f; ", .{i});
        }
        var acc_sum: []const u8 = "";
        for (0..nacc) |i| {
            acc_sum = acc_sum ++ std.fmt.comptimePrint("acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i });
        }
        return std.fmt.comptimePrint(
            \\auto sg = simdgroup_index_in_threadgroup;
            \\auto lane = thread_index_in_simdgroup;
            \\auto tg_n = threadgroup_position_in_grid.y;
            \\
            \\int K = int(K_size);
            \\int N = int(N_size);
            \\int K_by_p = K / 8;
            \\int K_by_gs = K / GS;
            \\int n0 = (int(tg_n) * NSG + int(sg)) * {d};
            \\if (n0 + {d} >= N) {{ return; }}
            \\
            \\float acc[{d}];
            \\{s}
            \\
            \\using Vec8 = vec<T, 8>;
            \\const device Vec8 *xv = (const device Vec8*)x;
            \\
            \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
            \\  int k_base = pack * 8;
            \\  int gi = k_base / GS;
            \\{s}
            \\}}
            \\
            \\{s}
            \\
            \\if (lane < {d}) {{
            \\  int j = int(lane) / {d};
            \\  int row = int(lane) - j * {d};
            \\  y[row * N + n0 + j] = T(acc[int(lane)]);
            \\}}
        , .{ bn, bn - 1, nacc, acc_init, body, acc_sum, nacc, m, m });
    }
}

/// msg column-tile width per M: 4 columns through M=6, 2 for M=7 (14
/// accumulators — under the 24 ceiling; covers the depth-6 verify lm_head).
fn vqmmMsgBn(m: c_int) c_int {
    return if (m <= 6) 4 else 2;
}

const VQMM_MSG_SOURCES = [6][:0]const u8{
    verifyQmmMsgSource(2, 4), verifyQmmMsgSource(3, 4), verifyQmmMsgSource(4, 4),
    verifyQmmMsgSource(5, 4), verifyQmmMsgSource(6, 4), verifyQmmMsgSource(7, 2),
};
const VQMM_MSG_NAMES = [6][*:0]const u8{
    "mlxserve_vqmm_msg_m2", "mlxserve_vqmm_msg_m3", "mlxserve_vqmm_msg_m4",
    "mlxserve_vqmm_msg_m5", "mlxserve_vqmm_msg_m6", "mlxserve_vqmm_msg_m7",
};

var vqmm_msg_kernels: [6]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmMsgKernel(m: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx: usize = @intCast(m - 2);
    if (vqmm_msg_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        VQMM_MSG_NAMES[idx],
        in_vec,
        out_vec,
        VQMM_MSG_SOURCES[idx],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_msg_kernels[idx] = kernel;
    return kernel;
}

/// Column-tile width per M: 4 columns through M=6 (24 accumulators, the
/// measured register ceiling), 2 columns for M=7 (14 accumulators) — one
/// deep-verify width past MTPLX's family (their contract caps depth at 3).
/// M=8 measured a CLIFF (T(7) round 636 ms vs 115 stock — spill/occupancy
/// past 7 live Vec8 row vectors), so 8+ falls back to stock qmm and the
/// adaptive depth cap keeps verify at seq <= 7.
fn vqmmBn(m: c_int) c_int {
    return if (m <= 6) 4 else 2;
}

const VQMM_SOURCES = [6][:0]const u8{
    verifyQmmSource(2, 4), verifyQmmSource(3, 4), verifyQmmSource(4, 4),
    verifyQmmSource(5, 4), verifyQmmSource(6, 4), verifyQmmSource(7, 2),
};
const VQMM_NAMES = [6][*:0]const u8{
    "mlxserve_vqmm_ks_m2", "mlxserve_vqmm_ks_m3", "mlxserve_vqmm_ks_m4",
    "mlxserve_vqmm_ks_m5", "mlxserve_vqmm_ks_m6", "mlxserve_vqmm_ks_m7",
};

var vqmm_kernels: [6]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmKernel(m: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx: usize = @intCast(m - 2);
    if (vqmm_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const source: [*:0]const u8 = VQMM_SOURCES[idx];
    const name: [*:0]const u8 = VQMM_NAMES[idx];
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_kernels[idx] = kernel;
    return kernel;
}

/// Process-lifetime cache of 0-d int scalars for the kernel's K/N inputs —
/// only a handful of distinct layer geometries exist per model, and the
/// verify path builds ~300 kernel nodes per round (inference thread only,
/// same single-caller discipline as the kernel caches above).
var vqmm_scalar_cache: [16]struct { v: c_int, arr: mlx.mlx_array } = undefined;
var vqmm_scalar_count: usize = 0;

fn cachedScalarInt(v: c_int) mlx.mlx_array {
    for (vqmm_scalar_cache[0..vqmm_scalar_count]) |e| {
        if (e.v == v) return e.arr;
    }
    const arr = mlx.mlx_array_new_int(v);
    if (vqmm_scalar_count < vqmm_scalar_cache.len) {
        vqmm_scalar_cache[vqmm_scalar_count] = .{ .v = v, .arr = arr };
        vqmm_scalar_count += 1;
    } else {
        // Evict slot 0 (safe: in-flight lazy graph nodes hold their own
        // refs to the old array; this only drops OUR handle).
        _ = mlx.mlx_array_free(vqmm_scalar_cache[0].arr);
        vqmm_scalar_cache[0] = .{ .v = v, .arr = arr };
    }
    return arr;
}

var verify_qmm_enabled_cache: ?bool = null;
pub fn verifyQmmEnabled() bool {
    if (verify_qmm_enabled_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_VERIFY_QMM")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    verify_qmm_enabled_cache = on;
    return on;
}

pub const VqmmLane = enum { none, splitk, msg, nax };

/// The byte-unpack NAX tile is profitable for oQe's wide q5/q6 projections,
/// but loses to stock MLX on the 1024-wide K/V projections. This threshold is
/// the measured split on M5 Max at M=8; q4 keeps its existing full geometry.
fn mixedNaxShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    return (bits == 5 or bits == 6) and group_size == 64 and n >= 5120;
}

/// Pure lane selection for the verify-qmm family (hermetically pinned by
/// the NAX dispatch-table test). Shared floors: M 2..16, N >= 512 (tiny-N
/// projections — GDN ba proj, MoE routers — gain nothing over stock qmv
/// while the per-call kernel-node build ~10us is real; ~50 such calls ride
/// every verify round). Lanes:
/// - nax: M >= nax_min_m (default 8, MLX_SERVE_VERIFY_QMM_NAX_MIN_M) when
///   the M5-class probe is live AND the m16 tile's stricter geometry holds
///   (K % 256 == 0, N % 32 == 0) — the lm_head N=151936 qualifies natively,
///   so no msg variant is needed past M=7.
/// - splitk/msg: the plain-SIMD M 2..7 lanes (K % 64 == 0, N % 4 == 0),
///   byte-identical to the pre-NAX dispatch; msg takes N >= 100000.
/// - none: stock mlx_quantized_matmul.
pub fn vqmmLaneFor(m: c_int, K: c_int, N: c_int, nax_on: bool, nax_min_m: c_int) VqmmLane {
    if (m < 2 or m > 16) return .none;
    if (N < 512) return .none;
    if (nax_on and m >= nax_min_m and @mod(K, 256) == 0 and @mod(N, 32) == 0) return .nax;
    if (m > 7) return .none;
    if (@mod(K, 64) != 0 or @mod(N, 4) != 0) return .none;
    return if (N >= 100000) .msg else .splitk;
}

/// Verify-width qmm dispatch (three lanes; see vqmmLaneFor). Returns the
/// [.., M, N] product when the shape is eligible — split-K (M 2..7),
/// msg wide tile (M 2..7 at huge N), or the NAX m16 tile (M 8..16,
/// M5-class machines only) — null to fall through to stock. 4-bit affine,
/// gs in {32,64,128}, bf16/fp16 activations.
pub fn verifyQmm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!verifyQmmEnabled()) return null;
    // The plain-SIMD split-K/msg kernels below remain 4-bit specializations.
    // The M5 NAX tile additionally handles the 5/6-bit affine projections in
    // oQe checkpoints; those widths fall through to stock below the NAX
    // takeover row instead of entering a 4-bit kernel.
    if (bits != 4 and bits != 5 and bits != 6) return null;
    if (group_size != 32 and group_size != 64 and group_size != 128) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return null;
    const K: c_int = xsh[xsh.len - 1];
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| m *= d;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const N: c_int = wsh[0];
    // Affine packed geometry sanity: one uint32 column carries 32/bits source
    // values (5/6-bit rows are byte-packed but still exposed as uint32 arrays).
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * @as(i64, bits)) return null;
    const nax_on = naxLaneEnvEnabled() and verifyQmmNaxAvailable();
    const lane = vqmmLaneFor(m, K, N, nax_on, naxMinM());
    if (bits != 4 and
        (lane != .nax or
            !naxMixedBitsEnvEnabled() or
            !mixedNaxShapeEnabled(bits, group_size, N))) return null;
    switch (lane) {
        .none => return null,
        // NAX m16 tile (M5-class): M 8..16 by default — past the plain-SIMD
        // register cliff. Construction stays strictly behind the probe.
        .nax => return try runVerifyQmmNax(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        // Huge-N (lm_head class): the tiny-tile split-K grid thrashes the
        // scheduler there (measured 2.1x stock at M=4) — route through the
        // wide multi-simdgroup tile instead (2-column tile at M=7).
        .msg => return try runVerifyQmmMsg(s, x, w, sc, bi, group_size, m, K, N, xd, xsh),
        .splitk => {},
    }

    // x rides in with its original shape — the kernel indexes the buffer
    // linearly, and a contiguous [.., M, K] has the same layout as [M, K]
    // (ensure_row_contiguous copies the rare non-contiguous case).
    const k_parts: c_int = if (N >= 4096) 2 else 4;
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * k_parts, @divExact(N, vqmmBn(m)), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * k_parts, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K_PARTS", k_parts));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmKernel(m);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    // Restore the caller's leading shape with N as the last dim.
    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

const VQMM_MSG_NSG: c_int = 8; // simdgroups per threadgroup (their sweep winner)

fn runVerifyQmmMsg(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    const cols: c_int = vqmmMsgBn(m) * VQMM_MSG_NSG;
    const tg_count: c_int = @divTrunc(N + cols - 1, cols); // in-kernel n0 guard covers the tail
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * VQMM_MSG_NSG, tg_count, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * VQMM_MSG_NSG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", VQMM_MSG_NSG));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmMsgKernel(m);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

// ── NAX m16 verify tile (M5-class matrix units; see the section comment
// above for provenance, gating, and the never-build-off-probe rule) ──

/// Case-insensitive prefix match on the M5-class GPU family identifier.
/// Prefix (not equality) is MTPLX's shipping behavior — device variants
/// report suffixed forms ("applegpu_g17s", "applegpu_g17d").
pub fn naxArchIsG17(arch: []const u8) bool {
    const prefix = "applegpu_g17";
    if (arch.len < prefix.len) return false;
    return std.ascii.eqlIgnoreCase(arch[0..prefix.len], prefix);
}

/// "26.4"/"26.4.1"-style product version at least req_major.req_minor.
/// Unparseable components read as 0 (mirrors MTPLX: int() failures fall
/// back to 0, so garbage can never satisfy the floor).
pub fn macosVersionAtLeast(ver: []const u8, req_major: u32, req_minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const minor: u32 = if (it.next()) |mn| (std.fmt.parseInt(u32, mn, 10) catch 0) else 0;
    return major > req_major or (major == req_major and minor >= req_minor);
}

/// The whole availability gate, pure over its inputs (mirror of MTPLX's
/// nax_available()): not force-fallback, G17-class GPU, macOS >= 26.2 (the
/// MetalPerformancePrimitives floor).
pub fn naxAvailableFrom(force_fallback: bool, arch: []const u8, os_ver: []const u8) bool {
    if (force_fallback) return false;
    if (!naxArchIsG17(arch)) return false;
    return macosVersionAtLeast(os_ver, 26, 2);
}

/// Human-readable NAX status for `--version` / the app's Settings ("nax"
/// line, first token on/off, remainder the reason). Hardware + OS is the
/// whole story for OUR binaries: the bundled MLX always ships the NAX
/// kernels (build-mlx.sh + tests/test_mlx_staged_nax.sh assert them), so
/// this mirrors MLX's stock-op is_nax_available() gate (GPU gen >= 17,
/// macOS >= 26.2 — that symbol is not exported, hence the mirror).
/// Deliberately ignores the verifyQmm-lane QA env switches: those scope
/// our custom kernel lane, not MLX's stock dispatch.
pub fn naxStatusFrom(arch: []const u8, os_ver: []const u8) []const u8 {
    if (!naxArchIsG17(arch)) return "off (requires M5-class GPU)";
    if (!macosVersionAtLeast(os_ver, 26, 2)) return "off (requires macOS 26.2+)";
    return "on (M5 neural accelerators)";
}

/// naxStatusFrom over the real device: mlx device-info arch + sysctl OS
/// version (same sources as the verifyQmm probe).
pub fn naxStatus() []const u8 {
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxStatusFrom(arch, ver);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// "kern.osproductversion" → "26.4"-style string (the sysctl mirror of
/// Python's platform.mac_ver()[0]).
fn macosProductVersion(buf: []u8) ?[]const u8 {
    var len: usize = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return null;
    var n = @min(len, buf.len);
    while (n > 0 and buf[n - 1] == 0) n -= 1;
    if (n == 0) return null;
    return buf[0..n];
}

/// GPU architecture identifier off mlx device info ("applegpu_g16" on the
/// M4 Max). The returned pointer from mlx is borrowed from the info object,
/// so the string is copied into the caller's buffer before the info frees.
fn gpuArchitecture(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

/// Test seam: force the NAX availability probe (null = real probe). Only
/// ever force FALSE on non-G17 machines — forcing true would let dispatch
/// build a kernel whose matmul2d pipeline cannot be created off M5-class
/// hardware.
pub var vqmm_nax_probe_override: ?bool = null;

var vqmm_nax_avail_cache: ?bool = null;

/// M5-class NAX units present: "applegpu_g17" GPU + macOS >= 26.2.
/// MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1 pretends the units are absent so
/// an M5 can rehearse the exact M1-M4 plain-SIMD path (QA switch, mirrored
/// from MTPLX). Cached for the process lifetime (inference-thread caller
/// discipline, same as the kernel caches).
pub fn verifyQmmNaxAvailable() bool {
    if (vqmm_nax_probe_override) |v| return v;
    if (vqmm_nax_avail_cache) |v| return v;
    const ok = computeNaxAvailable();
    vqmm_nax_avail_cache = ok;
    return ok;
}

fn computeNaxAvailable() bool {
    var force = false;
    if (std.c.getenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        force = v.len > 0 and v[0] == '1';
    }
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxAvailableFrom(force, arch, ver);
}

var vqmm_nax_env_cache: ?bool = null;
/// Lane kill switch: MLX_SERVE_VERIFY_QMM_NAX=0 (the family-wide
/// MLX_SERVE_VERIFY_QMM=0 also covers it via verifyQmm's entry gate).
fn naxLaneEnvEnabled() bool {
    if (vqmm_nax_env_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    vqmm_nax_env_cache = on;
    return on;
}

var vqmm_nax_mixed_env_cache: ?bool = null;
/// oQe A/B seam: keep the q4 NAX lane live while routing affine q5/q6 back
/// through stock qmm. This isolates the mixed-bit kernel contribution without
/// changing model/controller state or disabling NAX for q4 projections.
fn naxMixedBitsEnvEnabled() bool {
    if (vqmm_nax_mixed_env_cache) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX_MIXED")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    vqmm_nax_mixed_env_cache = on;
    return on;
}

/// Family-wide NAX readiness: family kill switch, lane kill switch, and the
/// hardware probe all pass. This intentionally says nothing about a specific
/// M/K/N dispatch; controller decisions use verifyQmmNaxEnabledForM instead.
pub fn verifyQmmNaxEnabled() bool {
    return verifyQmmEnabled() and naxLaneEnvEnabled() and verifyQmmNaxAvailable();
}

/// MLX_SERVE_VERIFY_QMM_NAX_MIN_M parse: the M width where the NAX tile
/// takes over from the plain-SIMD lanes. Default 8 (MTPLX's dispatcher
/// keeps SIMD through M<=6 even with NAX lit — 16-row padding waste makes
/// SIMD competitive at small M; our lanes cover 7). The M5-day A/B of
/// routing M 5..7 to NAX (todo-m5-nax.md §7 step 2) sets 5 — same boot, no
/// rebuild. Clamped to [2,16]; disabling is the kill switch's job.
pub fn naxMinMFrom(val: ?[]const u8) c_int {
    const v = val orelse return 8;
    const parsed = std.fmt.parseInt(c_int, v, 10) catch return 8;
    return @min(16, @max(2, parsed));
}

var vqmm_nax_min_m_cache: ?c_int = null;
fn naxMinM() c_int {
    if (vqmm_nax_min_m_cache) |v| return v;
    const m = naxMinMFrom(if (std.c.getenv("MLX_SERVE_VERIFY_QMM_NAX_MIN_M")) |p| std.mem.span(p) else null);
    vqmm_nax_min_m_cache = m;
    return m;
}

/// Exact NAX-dispatch predicate, pure over the runtime gates and shape. Keep
/// controller policy behind this seam so a kill switch, takeover-width tweak,
/// or ineligible projection geometry can never advertise a NAX cost profile
/// while verifyQmm would actually fall through to stock qmm.
pub fn verifyQmmNaxEnabledForMFrom(
    m: c_int,
    K: c_int,
    N: c_int,
    verify_on: bool,
    lane_on: bool,
    available: bool,
    min_m: c_int,
) bool {
    return verify_on and vqmmLaneFor(m, K, N, lane_on and available, min_m) == .nax;
}

/// Runtime form of verifyQmmNaxEnabledForMFrom.
pub fn verifyQmmNaxEnabledForM(m: c_int, K: c_int, N: c_int) bool {
    return verifyQmmNaxEnabledForMFrom(
        m,
        K,
        N,
        verifyQmmEnabled(),
        naxLaneEnvEnabled(),
        verifyQmmNaxAvailable(),
        naxMinM(),
    );
}

const VQMM_NAX_HEADER: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\
;

/// MTPLX's _build_kernel_m16_nax_ktmpl body VERBATIM (nax_verify.py,
/// Apache-2.0; single-brace form of their f-string). The tensor
/// extents/strides encode dflash's working matmul2d convention — treat
/// them as opaque and correct, do not re-derive. Geometry: threadgroup =
/// 256 threads = 8 simdgroups; each simdgroup owns a K/8 chunk and stages
/// a dequantized 16x32 B tile in threadgroup memory (each of its 32 lanes
/// dequants ONE output column's 16 K-values — two 4-bit packs — per
/// iteration); matmul2d (16x32x16 multiply_accumulate, execution_simdgroup)
/// multiplies the row-padded activation tile against it into a cooperative
/// fp32 accumulator; after the K loop the 8 partial C tiles reduce across
/// the threadgroup. T/GS/KCONST arrive as template args (MLX name-mangles
/// template values into the host_name, so one kernel object serves every
/// specialization — the GDN Dk/Dv precedent).
const VQMM_NAX_SOURCE: [:0]const u8 =
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
    \\constexpr int BM = 16;
    \\constexpr int BN = 32;
    \\constexpr int BK = 16;
    \\constexpr int NSG = 8;
    \\constexpr int K = KCONST;
    \\constexpr int K_bytes = K * BITS / 8;
    \\constexpr int K_by_gs = K / GS;
    \\constexpr int K_chunk = K / NSG;
    \\
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint sg_id = simdgroup_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint tg_n = threadgroup_position_in_grid.y;
    \\int N = int(N_size);
    \\int n0 = int(tg_n) * BN;
    \\int k_begin = int(sg_id) * K_chunk;
    \\int k_end = k_begin + K_chunk;
    \\
    \\threadgroup T B_tile[NSG][BK * BN];
    \\threadgroup float partial[NSG][BM * BN];
    \\
    \\constexpr auto desc = matmul2d_descriptor(
    \\    16,
    \\    32,
    \\    16,
    \\    false,
    \\    false,
    \\    false,
    \\    matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, metal::execution_simdgroup> op;
    \\
    \\tensor<device T, dextents<int, 2>, tensor_inline> A(
    \\    (device T*)x,
    \\    dextents<int, 2>{K, BM},
    \\    array<int, 2>{1, K});
    \\tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
    \\    B_tile[sg_id],
    \\    dextents<int, 2>{BN, BK},
    \\    array<int, 2>{1, BN});
    \\tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
    \\    partial[sg_id],
    \\    dextents<int, 2>{BN, BM},
    \\    array<int, 2>{1, BN});
    \\
    \\auto ct_c = op.template get_destination_cooperative_tensor<
    \\    tensor<device T, extents<int, 16, 16>, tensor_inline>,
    \\    tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
    \\    float>();
    \\_Pragma("unroll")
    \\for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    \\    ct_c[i] = 0.0f;
    \\}
    \\
    \\int n_global = n0 + int(lane);
    \\for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    \\    const device uchar* wp =
    \\        ((const device uchar*)w_q) + n_global * K_bytes + (k0 * BITS) / 8;
    \\    float scale = float(scales[n_global * K_by_gs + (k0 / GS)]);
    \\    float bias = float(biases[n_global * K_by_gs + (k0 / GS)]);
    \\
    \\    if constexpr (BITS == 4) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 4 + 0]) |
    \\                (uint32_t(wp[pack * 4 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 4 + 2]) << 16) |
    \\                (uint32_t(wp[pack * 4 + 3]) << 24);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = (p >> (ki * 4)) & 0xFu;
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 5) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            ulong p =
    \\                ulong(wp[pack * 5 + 0]) |
    \\                (ulong(wp[pack * 5 + 1]) << 8) |
    \\                (ulong(wp[pack * 5 + 2]) << 16) |
    \\                (ulong(wp[pack * 5 + 3]) << 24) |
    \\                (ulong(wp[pack * 5 + 4]) << 32);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = uint32_t((p >> (ki * 5)) & 0x1Ful);
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 4; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 3 + 0]) |
    \\                (uint32_t(wp[pack * 3 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 3 + 2]) << 16);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 4; ++ki) {
    \\                uint32_t q = (p >> (ki * 6)) & 0x3Fu;
    \\                B_tile[sg_id][(pack * 4 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    }
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    auto tA = A.template slice<16, 16>(k0, 0);
    \\    auto tB = B.template slice<32, 16>(0, 0);
    \\    op.run(tA, tB, ct_c);
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\auto tC = C.template slice<32, 16>(0, 0);
    \\ct_c.store(tC);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\for (int off = int(tid); off < BM * BN; off += NSG * 32) {
    \\    float acc01 = partial[0][off] + partial[1][off];
    \\    float acc23 = partial[2][off] + partial[3][off];
    \\    float acc45 = partial[4][off] + partial[5][off];
    \\    float acc67 = partial[6][off] + partial[7][off];
    \\    float acc = (acc01 + acc23) + (acc45 + acc67);
    \\    int row = off / BN;
    \\    int col = off - row * BN;
    \\    y[row * N + n0 + col] = T(acc);
    \\}
;

var vqmm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// ONLY call where verifyQmmNaxAvailable() — see the never-build rule.
fn getVerifyQmmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (vqmm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_vqmm_nax_m16",
        in_vec,
        out_vec,
        VQMM_NAX_SOURCE,
        VQMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_nax_kernel = kernel;
    return kernel;
}

/// The M-padding half of the NAX host contract, split out so it can be
/// exercised hermetically on non-M5 machines (the scaffolding test wraps
/// it around STOCK qmm — zero pad rows must produce zero output rows):
/// collapse the caller's leading shape to [m, K] (mirrors MTPLX's
/// x.reshape(m, k) so row padding is well-defined for every batch shape),
/// then zero-pad the row axis to the tile's fixed 16. Owned handle.
fn naxPadTo16(s: mlx.mlx_stream, x: mlx.mlx_array, m: c_int, K: c_int, xd: mlx.mlx_dtype) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    const x2_shape = [_]c_int{ m, K };
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2_shape, 2, s));
    var x16 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x16);
    if (m < 16) {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        const zdims = [_]c_int{};
        try mlx.check(mlx.mlx_zeros(&zero, &zdims, 0, xd, s));
        const pad_axes = [_]c_int{0};
        const pad_low = [_]c_int{0};
        const pad_high = [_]c_int{16 - m};
        try mlx.check(mlx.mlx_pad(&x16, x2, &pad_axes, 1, &pad_low, 1, &pad_high, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&x16, x2));
    }
    return x16;
}

/// The slice-back half: first m rows of the [16, N] tile output (pad rows
/// dropped). Owned handle.
fn naxSliceRows(s: mlx.mlx_stream, y16: mlx.mlx_array, m: c_int, N: c_int) !mlx.mlx_array {
    var ym = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ym);
    if (m < 16) {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ m, N };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&ym, y16, &start, 2, &stop, 2, &strides, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&ym, y16));
    }
    return ym;
}

/// Run the NAX m16 tile: pad the activations to the fixed [16, K] tile
/// ("weight streaming dominates so padded rows are nearly free" — the
/// source ledger), grid (256, N/32), slice y back to the caller's M rows.
fn runVerifyQmmNax(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    const x16 = try naxPadTo16(s, x, m, K, xd);
    defer _ = mlx.mlx_array_free(x16);

    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ 16, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(N, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KCONST", K));

    const inputs_arr = [_]mlx.mlx_array{ x16, w, sc, bi, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmNaxKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y16);
    try mlx.check(mlx.mlx_vector_array_get(&y16, outputs_vec, 0));

    // Drop the pad rows, restore the caller's leading shape with N last.
    const ym = try naxSliceRows(s, y16, m, N);
    defer _ = mlx.mlx_array_free(ym);

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, ym, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

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
// contiguous — guaranteed for slices along T). Kill switch:
// MLX_SERVE_FUSED_256=0 restores the composed path AND the old guard
// budgeting (one shared predicate, so guards and dispatch cannot drift).
const ATTN256_KERNEL_HEADER =
    \\#include <metal_simdgroup_matrix>
    \\
    \\// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
    \\// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
    \\// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
    \\// holding one row differ in lane bits 0 and 3 (see msv_coord).
    \\inline short2 msv_coord(ushort lane) {
    \\  const short qid = lane / 4;
    \\  const short fm = (qid & 4) + ((lane / 2) % 4);
    \\  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    \\  return short2(fn, fm);
    \\}
    \\
    \\inline void msv_mma(thread float2 &d, float2 a, float2 b) {
    \\  metal::simdgroup_float8x8 D, A, B, C;
    \\  A.thread_elements()[0] = a.x;
    \\  A.thread_elements()[1] = a.y;
    \\  B.thread_elements()[0] = b.x;
    \\  B.thread_elements()[1] = b.y;
    \\  C.thread_elements()[0] = d.x;
    \\  C.thread_elements()[1] = d.y;
    \\  simdgroup_multiply_accumulate(D, A, B, C);
    \\  d.x = D.thread_elements()[0];
    \\  d.y = D.thread_elements()[1];
    \\}
    \\
    \\inline float msv_row_max(float2 v) {
    \\  float t = metal::max(v.x, v.y);
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
    \\  return t;
    \\}
    \\
    \\inline float msv_row_sum(float2 v) {
    \\  float t = v.x + v.y;
    \\  t += metal::simd_shuffle_xor(t, ushort(1));
    \\  t += metal::simd_shuffle_xor(t, ushort(8));
    \\  return t;
    \\}
    \\
;

// v2 tile geometry (2026-07-12 micro-bench sweep on the M4 Max, Qwen 24q/4kv
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
const ATTN256_KERNEL_SOURCE =
    \\constexpr int BQ = 64;
    \\constexpr int BK = 32;
    \\constexpr int BD = 256;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = BD + 8;
    \\constexpr int NT = 256;
    \\
    \\const int qL = q_shape[2];
    \\const int kL = k_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = k_shape[1];
    \\const int gqa = Hq / Hk;
    \\
    \\const int tqx = int(threadgroup_position_in_grid.x);
    \\const int hq = int(threadgroup_position_in_grid.y);
    \\const int bb = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\// exp2-based softmax (steel convention): fold log2(e) into the scale.
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\const int SW = win[0];
    \\
    \\// Budgeted kv-chunk dispatch (causal arm): this dispatch covers keys
    \\// [k_begin, k_end); online-softmax state (m/l/unnormalized O) rides fp32
    \\// carry buffers between dispatches — exact register precision, so the
    \\// chunked chain is bit-identical to one long dispatch. phase bit 0 =
    \\// carry-in present, bit 1 = final chunk (normalize + bf16 store).
    \\const int k_begin = kr[0];
    \\const int k_end = metal::min(kr[1], kL);
    \\const bool has_carry = (phase[0] & 1) != 0;
    \\const bool is_final = (phase[0] & 2) != 0;
    \\
    \\// Bottom-right aligned causal: query row r of this chunk sits at
    \\// absolute KV position q_off + r.
    \\const int q_off = kL - qL;
    \\
    \\const device T* Qp = q + bb * q_strides[0] + hq * q_strides[1]
    \\    + (long)(tqx * BQ) * q_strides[2];
    \\const device T* Kp = k + bb * k_strides[0] + (hq / gqa) * k_strides[1];
    \\const device T* Vp = v + bb * v_strides[0] + (hq / gqa) * v_strides[1];
    \\device T* Op = out + (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ)) * BD;
    \\
    \\// K^T tile [BD][LDK] (20.5 KB) is strictly larger than the V tile
    \\// [BK][LDV] (16.9 KB) — share one buffer, steel-style (barriers separate
    \\// the K reads from the V staging).
    \\threadgroup T KVs[LDK * BD];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const int q_rows = metal::min(BQ, qL - tqx * BQ);
    \\
    \\const short2 sc = msv_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\
    \\// Q fragment straight from global into registers, once per threadgroup:
    \\// row (tm+sm), element pairs (dd*8+sn, +1). sn is always even, so the
    \\// vec2 load is 4B-aligned for any row whose stride is even (structural:
    \\// q rows stride 256). Rows past the ragged tail read zero — their
    \\// outputs are discarded by the store guard.
    \\float2 Qfrag[BD / 8];
    \\{
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const device T* Qrow = Qp + (long)qr * q_strides[2];
    \\    for (int dd = 0; dd < BD / 8; ++dd) {
    \\      const vec<T, 2> p = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\      Qfrag[dd] = float2(float(p.x), float(p.y));
    \\    }
    \\  } else {
    \\    for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
    \\  }
    \\}
    \\
    \\float2 Ofrag[BD / 8];
    \\for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
    \\// Init max FINITE (not -inf) so the rescale factor exp2(old-new) can
    \\// never be exp2(nan); masked scores use true -INFINITY so a row whose
    \\// first blocks are fully banded out contributes exp2(-inf)=0, not 1.
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\// Resume carried softmax state from the previous kv chunk. Rows past
    \\// the ragged q tail never carry (their outputs are discarded anyway).
    \\// (m_in/l_in/o_in are indexed directly, never through a typed pointer:
    \\// MLX binds sub-4KB inputs — the no-carry dummies — in the `constant`
    \\// address space and real carries in `device`, and the source must
    \\// compile under both.)
    \\if (has_carry) {
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + qr));
    \\    max_score = m_in[crow];
    \\    sum_score = l_in[crow];
    \\    const long obase = crow * (long)BD;
    \\    for (int dd = 0; dd < BD / 8; ++dd) {
    \\      Ofrag[dd] = float2(o_in[obase + dd * 8 + sn], o_in[obase + dd * 8 + sn + 1]);
    \\    }
    \\  }
    \\}
    \\
    \\const int NK = (k_end + BK - 1) / BK;
    \\const int q_lo = tqx * BQ + q_off;
    \\const int q_hi = q_lo + BQ - 1;
    \\const int kb_lim = metal::min(NK, (q_hi + BK) / BK);
    \\int kb = k_begin / BK;
    \\if (SW > 0) kb = metal::max(kb, metal::max(0, q_lo - SW + 1) / BK);
    \\const int kb_min_causal = metal::max(0, q_lo) / BK;
    \\const int row_pos = q_lo + tm + sm;
    \\
    \\for (; kb < kb_lim; kb++) {
    \\  const int c0 = kb * BK;
    \\  const int rows_k = metal::min(BK, kL - c0);
    \\
    \\  // Stage K transposed (Ks[d][kk]): uint4 global loads (8 bf16 each),
    \\  // scalar transposed scatter into smem.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Kp + (long)(c0 + r) * k_strides[2]) + c8);
    \\    }
    \\    thread T* e = (thread T*)&w;
    \\    const int cb = c8 * 8;
    \\    Ks[(cb + 0) * LDK + r] = e[0];
    \\    Ks[(cb + 1) * LDK + r] = e[1];
    \\    Ks[(cb + 2) * LDK + r] = e[2];
    \\    Ks[(cb + 3) * LDK + r] = e[3];
    \\    Ks[(cb + 4) * LDK + r] = e[4];
    \\    Ks[(cb + 5) * LDK + r] = e[5];
    \\    Ks[(cb + 6) * LDK + r] = e[6];
    \\    Ks[(cb + 7) * LDK + r] = e[7];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // S = Q @ K^T for this simdgroup's 8 query rows.
    \\  float2 Sfrag[BK / 8];
    \\  for (int i = 0; i < BK / 8; ++i) Sfrag[i] = float2(0.0f);
    \\  for (int dd = 0; dd < BD / 8; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    const float2 kf0 = float2(float(Ks[kbase]), float(Ks[kbase + 1]));
    \\    const float2 kf1 = float2(float(Ks[kbase + 8]), float(Ks[kbase + 9]));
    \\    const float2 kf2 = float2(float(Ks[kbase + 16]), float(Ks[kbase + 17]));
    \\    const float2 kf3 = float2(float(Ks[kbase + 24]), float(Ks[kbase + 25]));
    \\    msv_mma(Sfrag[0], qf, kf0);
    \\    msv_mma(Sfrag[1], qf, kf1);
    \\    msv_mma(Sfrag[2], qf, kf2);
    \\    msv_mma(Sfrag[3], qf, kf3);
    \\  }
    \\  Sfrag[0] *= scale_log2e;
    \\  Sfrag[1] *= scale_log2e;
    \\  Sfrag[2] *= scale_log2e;
    \\  Sfrag[3] *= scale_log2e;
    \\
    \\  // Masking: kL remainder + causal + sliding band, all element-wise.
    \\  const bool tail_k = (rows_k < BK);
    \\  const bool need_causal = (kb >= kb_min_causal);
    \\  const bool need_band = (SW > 0) && (c0 <= q_hi - SW);
    \\  if (tail_k || need_causal || need_band) {
    \\    for (int kt = 0; kt < BK / 8; ++kt) {
    \\      for (int jj = 0; jj < 2; ++jj) {
    \\        const int col = c0 + kt * 8 + sn + jj;
    \\        bool masked = false;
    \\        if (tail_k && col >= kL) masked = true;
    \\        if (need_causal && row_pos < col) masked = true;
    \\        if (need_band && (row_pos - col) >= SW) masked = true;
    \\        if (masked) Sfrag[kt][jj] = -INFINITY;
    \\      }
    \\    }
    \\  }
    \\
    \\  // Stage V (same smem as K — K reads are done): uint4 on both sides.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Vp + (long)(c0 + r) * v_strides[2]) + c8);
    \\    }
    \\    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
    \\  }
    \\
    \\  // Online softmax (registers only, overlaps the V staging above).
    \\  float new_max = max_score;
    \\  new_max = metal::max(new_max, msv_row_max(Sfrag[0]));
    \\  new_max = metal::max(new_max, msv_row_max(Sfrag[1]));
    \\  new_max = metal::max(new_max, msv_row_max(Sfrag[2]));
    \\  new_max = metal::max(new_max, msv_row_max(Sfrag[3]));
    \\  Sfrag[0] = metal::exp2(Sfrag[0] - new_max);
    \\  Sfrag[1] = metal::exp2(Sfrag[1] - new_max);
    \\  Sfrag[2] = metal::exp2(Sfrag[2] - new_max);
    \\  Sfrag[3] = metal::exp2(Sfrag[3] - new_max);
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  const float rowsum = msv_row_sum(Sfrag[0]) + msv_row_sum(Sfrag[1])
    \\      + msv_row_sum(Sfrag[2]) + msv_row_sum(Sfrag[3]);
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // O += P @ V.
    \\  for (int id = 0; id < BD / 8; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    const float2 vf0 = float2(float(Vs[vbase]), float(Vs[vbase + 1]));
    \\    const float2 vf1 = float2(float(Vs[vbase + 8 * LDV]),
    \\                              float(Vs[vbase + 8 * LDV + 1]));
    \\    const float2 vf2 = float2(float(Vs[vbase + 16 * LDV]),
    \\                              float(Vs[vbase + 16 * LDV + 1]));
    \\    const float2 vf3 = float2(float(Vs[vbase + 24 * LDV]),
    \\                              float(Vs[vbase + 24 * LDV + 1]));
    \\    msv_mma(Ofrag[id], Sfrag[0], vf0);
    \\    msv_mma(Ofrag[id], Sfrag[1], vf1);
    \\    msv_mma(Ofrag[id], Sfrag[2], vf2);
    \\    msv_mma(Ofrag[id], Sfrag[3], vf3);
    \\  }
    \\}
    \\
    \\// Final chunk: normalize + store bf16 (output freshly allocated,
    \\// contiguous). Mid chunk: store the raw fp32 state for the next
    \\// dispatch (m/l written by all 4 lane-threads of a row — same value).
    \\const int local_row = tm + sm;
    \\if (local_row < q_rows) {
    \\  if (is_final) {
    \\    const float inv = 1.0f / sum_score;
    \\    device T* Optr = Op + (long)local_row * BD + sn;
    \\    for (int id = 0; id < BD / 8; ++id) {
    \\      Optr[id * 8] = T(Ofrag[id].x * inv);
    \\      Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
    \\    }
    \\  } else {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + local_row));
    \\    m_out[crow] = max_score;
    \\    l_out[crow] = sum_score;
    \\    device float* Orow = o_out + crow * (long)BD;
    \\    for (int id = 0; id < BD / 8; ++id) {
    \\      Orow[id * 8 + sn] = Ofrag[id].x;
    \\      Orow[id * 8 + sn + 1] = Ofrag[id].y;
    \\    }
    \\  }
    \\}
;

var attn256_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getAttn256Kernel() !mlx.mlx_fast_metal_kernel {
    if (attn256_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in" };
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

/// Kill switch (MLX_SERVE_FUSED_256=0 disables the kernel entirely). Test
/// seam: `fused256_override` forces BOTH arms on/off without the environment.
pub var fused256_override: ?bool = null;
var fused256_env_cached: ?bool = null;
var fused256_causal_env_cached: ?Fused256CausalMode = null;

pub fn fused256Enabled() bool {
    if (fused256_override) |v| return v;
    if (fused256_env_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_FUSED_256");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    fused256_env_cached = enabled;
    return enabled;
}

/// Plain-CAUSAL dispatch mode. Default `.all` since the kv-chunk dispatch
/// budget landed (2026-07-22): the historical net-LOSS of every pre-budget
/// ratio-gated variant (8K: fused-to-ratio-4 231.5, fused-to-ratio-2 232.9,
/// composed 234.7 tok/s — monotonic: more fused = slower, despite the v2
/// kernel winning the µbench at kL <= 2*qL) was the macOS IOGPU
/// interactivity-preemption class: one long dispatch scanning the whole KV
/// gets preempted, and the penalty scales with dispatch length. With the
/// key axis split into budget-sized dispatches (see
/// FUSED256_DEFAULT_DISPATCH_BUDGET) the causal arm WINS the same
/// same-session A/B on the 27B: +2.9%/+2.3%/+4.6% at 8K/16K/32K.
/// MLX_SERVE_FUSED_256_CAUSAL=0 restores composed causal (and the old
/// OOM-guard score billing via prefillHeadDimFused).
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
// MLX_SERVE_FUSED_256_BUDGET overrides (0 = single-dispatch, pre-budget
// behavior); default mirrors oMLX's 250M fallback (~23 ms/dispatch on the
// M4 Max at the kernel's measured ~1.1e10 work-units/s).
pub const FUSED256_DEFAULT_DISPATCH_BUDGET: i64 = 250_000_000;

/// Test seam: forces the budget without the environment.
pub var fused256_budget_override: ?i64 = null;
var fused256_budget_env_cached: ?i64 = null;

pub fn fused256DispatchBudget() i64 {
    if (fused256_budget_override) |v| return v;
    if (fused256_budget_env_cached) |v| return v;
    const v: i64 = blk: {
        const raw = std.c.getenv("MLX_SERVE_FUSED_256_BUDGET") orelse break :blk FUSED256_DEFAULT_DISPATCH_BUDGET;
        break :blk std.fmt.parseInt(i64, std.mem.sliceTo(raw, 0), 10) catch FUSED256_DEFAULT_DISPATCH_BUDGET;
    };
    fused256_budget_env_cached = v;
    return v;
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
    if (!fused256Enabled()) return .off;
    if (fused256_causal_env_cached) |v| return v;
    const mode: Fused256CausalMode = blk: {
        const raw = std.c.getenv("MLX_SERVE_FUSED_256_CAUSAL") orelse break :blk .all;
        if (std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0")) break :blk .off;
        break :blk .all;
    };
    fused256_causal_env_cached = mode;
    return mode;
}

/// ONE predicate consumed by the three prefill OOM guards AND the dispatch
/// sites: true when prefill attention at this head_dim will NOT materialize
/// the composed [heads, chunk, total_kv] score tensor — either MLX's own
/// fused kernel covers it (<= 128) or our hd-256 kernel does for EVERY arm.
/// Keyed on causal mode: at `.all` (the default since the budgeted-dispatch
/// flip) no hd-256 score tensor materializes; MLX_SERVE_FUSED_256_CAUSAL=0
/// restores composed causal AND the guards' score billing together. Guards
/// and dispatch must never drift (the effectivePrefillChunk rule).
pub fn prefillHeadDimFused(head_dim: u32) bool {
    return head_dim <= 128 or (head_dim == 256 and fused256CausalMode() == .all);
}

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
        if (fused256CausalMode() == .off) return null;
    }
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256) return null;
    // Short sequences (< 16: decode AND spec-decode VERIFY forwards) belong
    // to MLX's sdpa_vector, which covers hd 256 natively and beats a 64-row
    // prefill tile walking the whole KV for a few-row query — dispatching
    // those here measured decode 48 -> 18 tok/s at 4K ctx on the 27B (MTP
    // verify is seq 1+depth, up to 9 at the NAX depth-8 cap — with causal
    // fused default-on the floor must clear it). 16 matches oMLX's
    // _MIN_ROUTE_Q_LEN; genuine prefill chunks are always far above it.
    if (qs[2] < 16) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;

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
            const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
            const oc_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .bfloat16));
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
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));

        const inputs_arr = [_]mlx.mlx_array{
            q,                                k,
            v,                                scl,
            win,                              kr,
            phase,                            if (has_carry) m_prev else dummy,
            if (has_carry) l_prev else dummy, if (has_carry) o_prev else dummy,
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
            try mlx.check(mlx.mlx_vector_array_get(&m_new, outputs_vec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&l_new, outputs_vec, 2));
            try mlx.check(mlx.mlx_vector_array_get(&o_new, outputs_vec, 3));
            if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
            if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
            if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
            m_prev = m_new;
            l_prev = l_new;
            o_prev = o_new;
        }
    }
    fused256_last_dispatch_count = dispatches;
    return out;
}
const model_mod = @import("model.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;
const Weights = model_mod.Weights;

// ── KV Cache (standard attention) ──

pub const KVCacheEntry = struct {
    // Storage. In `off` (dense bf16) mode `keys`/`values` are the full
    // [B,H,T,D] buffers and the `*_scales`/`*_biases` fields stay null. In
    // `affine` mode `keys`/`values` hold packed uint32 codes
    // ([B,H,T, D*bits/32]) and the matching scales/biases hold
    // [B,H,T, D/group_size] bf16. Switched on `KVCache.config.scheme`.
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    keys_scales: mlx.mlx_array,
    keys_biases: mlx.mlx_array,
    values_scales: mlx.mlx_array,
    values_biases: mlx.mlx_array,

    // Views: same layout as the storage fields above but trimmed to
    // [..., offset, ...] (or last `sw` entries during sliding-window decode).
    // SDPA reads dense arrays via `KVCache.denseView`; in `off` mode the
    // dense pair aliases `key_view`/`value_view`, in `affine` mode the
    // dense pair is freshly dequantized from these triples on read.
    key_view: mlx.mlx_array,
    value_view: mlx.mlx_array,
    key_scales_view: mlx.mlx_array,
    key_biases_view: mlx.mlx_array,
    value_scales_view: mlx.mlx_array,
    value_biases_view: mlx.mlx_array,

    offset: usize, // logical token count (may be < buffer capacity)
    initialized: bool,
};

/// Materialized dense `[B,H,T,D]` K/V pair handed to SDPA. Owns its arrays
/// only when `owned == true` (i.e. when the cache stores quantized data and
/// `KVCache.denseView` had to dequantize on read). In dense mode `k`/`v`
/// alias the cache's `key_view`/`value_view` and `deinit` is a no-op.
///
/// Phase 2 (fused-attn): in `.affine` mode the view ALSO carries borrowed
/// references to the cache's quantized K/V triples (`k_triple_q`, etc.).
/// SDPA call sites that opt into the fused path (`ctx.kv_attn_fused`) read
/// the triple via `quantTriple()`; everyone else uses `k`/`v` as before
/// and pays for the dense materialization. The arrays are non-owning
/// borrows of the cache's `key_view` / `key_scales_view` / `key_biases_view`
/// (and the V trio) — the cache keeps them alive for the request's lifetime.
pub const DenseKVView = struct {
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    owned: bool,

    /// Borrowed quant triples. Set to `.ctx = null` when not applicable
    /// (scheme == .off, or scheme is a TurboQuant variant — those need
    /// the rotation undo step which the v1 fused path doesn't implement).
    /// Read-only; the cache owns these handles.
    k_triple_q: mlx.mlx_array = .{ .ctx = null },
    k_triple_scales: mlx.mlx_array = .{ .ctx = null },
    k_triple_biases: mlx.mlx_array = .{ .ctx = null },
    v_triple_q: mlx.mlx_array = .{ .ctx = null },
    v_triple_scales: mlx.mlx_array = .{ .ctx = null },
    v_triple_biases: mlx.mlx_array = .{ .ctx = null },
    /// True iff the triple fields above are populated. Lets call sites
    /// avoid checking `.ctx == null` on every field.
    has_quant_triple: bool = false,
    /// Quant params copied off the cache config so call sites don't need
    /// a pointer to it.
    bits: u8 = 0,
    group_size: u32 = 0,

    pub fn deinit(self: *DenseKVView) void {
        if (self.owned) {
            _ = mlx.mlx_array_free(self.k);
            _ = mlx.mlx_array_free(self.v);
            self.k = mlx.mlx_array_new();
            self.v = mlx.mlx_array_new();
            self.owned = false;
        }
        // Triple fields are non-owning borrows — never free.
    }

    pub fn kTriple(self: DenseKVView) kv_quant.BorrowedTriple {
        return .{ .q = self.k_triple_q, .scales = self.k_triple_scales, .biases = self.k_triple_biases };
    }
    pub fn vTriple(self: DenseKVView) kv_quant.BorrowedTriple {
        return .{ .q = self.v_triple_q, .scales = self.v_triple_scales, .biases = self.v_triple_biases };
    }
};

fn newEmptyKVEntry() KVCacheEntry {
    return .{
        .keys = mlx.mlx_array_new(),
        .values = mlx.mlx_array_new(),
        .keys_scales = mlx.mlx_array_new(),
        .keys_biases = mlx.mlx_array_new(),
        .values_scales = mlx.mlx_array_new(),
        .values_biases = mlx.mlx_array_new(),
        .key_view = mlx.mlx_array_new(),
        .value_view = mlx.mlx_array_new(),
        .key_scales_view = mlx.mlx_array_new(),
        .key_biases_view = mlx.mlx_array_new(),
        .value_scales_view = mlx.mlx_array_new(),
        .value_biases_view = mlx.mlx_array_new(),
        .offset = 0,
        .initialized = false,
    };
}

/// Reset a cache entry to the empty state, freeing all storage + view
/// handles. Mirror of the per-entry reset in `KVCache.restore`; used by the
/// disk tier (kv_disk_cache.zig) when installing restored buffers.
pub fn resetKVEntry(e: *KVCacheEntry) void {
    freeKVEntry(e);
    e.* = newEmptyKVEntry();
}

fn freeKVEntry(e: *KVCacheEntry) void {
    _ = mlx.mlx_array_free(e.keys);
    _ = mlx.mlx_array_free(e.values);
    _ = mlx.mlx_array_free(e.keys_scales);
    _ = mlx.mlx_array_free(e.keys_biases);
    _ = mlx.mlx_array_free(e.values_scales);
    _ = mlx.mlx_array_free(e.values_biases);
    _ = mlx.mlx_array_free(e.key_view);
    _ = mlx.mlx_array_free(e.value_view);
    _ = mlx.mlx_array_free(e.key_scales_view);
    _ = mlx.mlx_array_free(e.key_biases_view);
    _ = mlx.mlx_array_free(e.value_scales_view);
    _ = mlx.mlx_array_free(e.value_biases_view);
}

pub const KVCache = struct {
    entries: []KVCacheEntry,
    step: usize, // absolute sequence position (not affected by sliding window trimming)
    allocator: std.mem.Allocator,
    config: KVQuantConfig,
    /// Wave 2 — per-cache rotation matrices for the TurboQuant schemes.
    /// `null` for `off` and `affine`. Built once at `initWithConfig` time
    /// when the scheme is `turboquant_*`; reused across all updates. Lives
    /// on the cache so `snapshot`/`restore` can refcount-share through it
    /// (immutable post-init, safe to alias across snapshots).
    quant_state: ?kv_quant.TurboState,

    pub fn init(allocator: std.mem.Allocator, num_layers: u32) !KVCache {
        return initWithConfig(allocator, num_layers, KVQuantConfig.dense);
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, num_layers: u32, config: KVQuantConfig) !KVCache {
        return initWithConfigAndHeadDim(allocator, num_layers, config, 0);
    }

    /// TurboQuant schemes need a per-layer rotation-matrix slot. The actual
    /// matrix dimension isn't known yet — Gemma 4's cached K is at
    /// `2 * head_dim`, some archs differ per layer or between K/V — so we
    /// allocate empty slots here and `updateTurboQuant` lazy-builds the real
    /// matrix from the observed K/V last-dim on first write. `head_dim` is
    /// accepted but only used to fail-fast on obviously-bad configs.
    pub fn initWithConfigAndHeadDim(allocator: std.mem.Allocator, num_layers: u32, config: KVQuantConfig, head_dim: u32) !KVCache {
        const entries = try allocator.alloc(KVCacheEntry, num_layers);
        errdefer allocator.free(entries);
        for (entries) |*e| {
            e.* = newEmptyKVEntry();
        }
        var qs: ?kv_quant.TurboState = null;
        switch (config.scheme) {
            .turboquant_2, .turboquant_4 => {
                _ = head_dim; // observed at first write
                qs = try kv_quant.TurboState.initLazy(allocator, num_layers);
            },
            else => {},
        }
        return .{ .entries = entries, .step = 0, .allocator = allocator, .config = config, .quant_state = qs };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
        if (self.quant_state) |*qs| qs.deinit();
        self.quant_state = null;
    }

    /// Capture cache state for speculative-decoding rollback (PLD/drafter).
    /// Snapshots own array handles that share the underlying buffer with the
    /// source via refcount — cheap (no data copy) and immune to subsequent
    /// `update()` calls (which create new buffer handles when growing).
    /// `*_view` fields are excluded because `update()` recreates them every
    /// call.
    pub fn snapshot(self: *const KVCache) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        for (self.entries, 0..) |src, i| {
            out[i] = newEmptyKVEntry();
            out[i].offset = src.offset;
            out[i].initialized = src.initialized;
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&out[i].keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&out[i].values, src.values));
                if (self.config.scheme != .off) {
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_biases, src.values_biases));
                }
            }
        }
        return .{ .entries = out, .step = self.step, .allocator = self.allocator, .config = self.config };
    }

    /// Replace cache state with `snap`. Frees current entries' arrays first;
    /// re-binds via refcount-share from snapshot. After restore, the next
    /// `update()` will recreate `*_view` fields from the restored buffers.
    pub fn restore(self: *KVCache, snap: *const KVCacheSnapshot) !void {
        std.debug.assert(self.entries.len == snap.entries.len);
        for (self.entries, snap.entries) |*dst, src| {
            freeKVEntry(dst);
            dst.* = newEmptyKVEntry();
            dst.offset = src.offset;
            dst.initialized = src.initialized;
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&dst.keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&dst.values, src.values));
                if (self.config.scheme != .off) {
                    try mlx.check(mlx.mlx_array_set(&dst.keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&dst.keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&dst.values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&dst.values_biases, src.values_biases));
                }
            }
        }
        self.step = snap.step;
    }

    const chunk_step = 256;

    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        switch (self.config.scheme) {
            .off => return self.updateDense(layer, new_k, new_v, s, max_seq),
            .affine => return self.updateAffine(layer, new_k, new_v, s, max_seq),
            .turboquant_2, .turboquant_4 => return self.updateTurboQuant(layer, new_k, new_v, s, max_seq),
        }
    }

    /// Wave 2 — TurboQuant write path. Rotate K and V by the per-layer
    /// Hadamard matrices, then re-use the affine grow/write/view machinery.
    /// Read-back at SDPA time dequantizes + rotates back via `denseView`.
    fn updateTurboQuant(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const qs = if (self.quant_state) |*q| q else return error.MissingTurboState;
        // Lazy-init: observe the actual K and V last-dims from the incoming
        // tensors. Gemma 4 stores K at 2x head_dim; some archs split K/V
        // dims; lazy construction sidesteps all of that.
        const k_shape = mlx.getShape(new_k);
        const v_shape = mlx.getShape(new_v);
        const k_n: u32 = @intCast(k_shape[k_shape.len - 1]);
        const v_n: u32 = @intCast(v_shape[v_shape.len - 1]);
        const rk = try qs.ensureKLayer(s, layer, k_n);
        const rv = try qs.ensureVLayer(s, layer, v_n);

        // Rotate inputs along the last axis. Free as soon as the quantize
        // call produces the affine triples — those become the stored cache
        // contents.
        const rotated_k = try kv_quant.rotateLastDim(s, new_k, rk);
        defer _ = mlx.mlx_array_free(rotated_k);
        const rotated_v = try kv_quant.rotateLastDim(s, new_v, rv);
        defer _ = mlx.mlx_array_free(rotated_v);

        // Hand off to the affine writer for the grow/slice_update/view work.
        // The dense view it returns is rotated K/V — undo the rotation
        // before handing back to SDPA. We can't call updateAffine directly
        // because it dequantizes-without-rotate at the end; emit a thin
        // helper that returns the rotated views and we rotate-back here.
        const rotated_view = try self.updateAffineRotated(layer, rotated_k, rotated_v, s, max_seq);

        // Now rotate the dense view back to the original basis for SDPA.
        var dense_k = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense_k);
        try mlx.check(mlx.mlx_matmul(&dense_k, rotated_view.k, rk, s));
        var dense_v = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense_v);
        try mlx.check(mlx.mlx_matmul(&dense_v, rotated_view.v, rv, s));

        // Free the temporary rotated-basis dense view; we own a fresh one.
        var rv_mut = rotated_view;
        rv_mut.deinit();
        return .{ .k = dense_k, .v = dense_v, .owned = true };
    }

    /// Variant of `updateAffine` that returns the rotated-basis dense view
    /// instead of an unrotated one. Only called from `updateTurboQuant`,
    /// which rotates the result back before handing to SDPA.
    fn updateAffineRotated(self: *KVCache, layer: u32, rk_in: mlx.mlx_array, rv_in: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        return self.updateAffine(layer, rk_in, rv_in, s, max_seq);
    }

    fn updateAffine(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];
        const cfg = self.config;
        const group_size: u32 = cfg.group_size;
        const bits: u8 = cfg.bits;

        // 1. Free stale views (6 of them — dense + 4 quant scale/bias views).
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);
        _ = mlx.mlx_array_free(entry.key_scales_view);
        _ = mlx.mlx_array_free(entry.key_biases_view);
        _ = mlx.mlx_array_free(entry.value_scales_view);
        _ = mlx.mlx_array_free(entry.value_biases_view);
        entry.key_view = mlx.mlx_array_new();
        entry.value_view = mlx.mlx_array_new();
        entry.key_scales_view = mlx.mlx_array_new();
        entry.key_biases_view = mlx.mlx_array_new();
        entry.value_scales_view = mlx.mlx_array_new();
        entry.value_biases_view = mlx.mlx_array_new();

        // 2. Quantize incoming K/V.
        var new_kq = try kv_quant.quantizeAffine(s, new_k, group_size, bits);
        defer new_kq.deinit();
        var new_vq = try kv_quant.quantizeAffine(s, new_v, group_size, bits);
        defer new_vq.deinit();

        // 3. Shape info: new_k is [B, heads, new_len, head_dim].
        const new_shape = mlx.getShape(new_k);
        const new_len: usize = @intCast(new_shape[2]);
        const B = new_shape[0];
        const heads = new_shape[1];
        const head_dim_u32: u32 = @intCast(new_shape[3]);
        const q_last: c_int = @intCast(head_dim_u32 * @as(u32, bits) / 32);
        const sc_last: c_int = @intCast(head_dim_u32 / group_size);

        // 4. Grow buffers if needed (6 of them, in lockstep on the seq axis).
        if (!entry.initialized or entry.offset + new_len > bufferCapacity(entry.keys)) {
            const needed = entry.offset + new_len;
            const n_chunks = (needed + chunk_step - 1) / chunk_step;
            const new_cap: c_int = @intCast(n_chunks * chunk_step);

            try growQuantBuf(s, &entry.keys, entry.initialized, entry.offset, new_cap, B, heads, q_last, .uint32);
            try growQuantBuf(s, &entry.values, entry.initialized, entry.offset, new_cap, B, heads, q_last, .uint32);
            try growQuantBuf(s, &entry.keys_scales, entry.initialized, entry.offset, new_cap, B, heads, sc_last, .bfloat16);
            try growQuantBuf(s, &entry.keys_biases, entry.initialized, entry.offset, new_cap, B, heads, sc_last, .bfloat16);
            try growQuantBuf(s, &entry.values_scales, entry.initialized, entry.offset, new_cap, B, heads, sc_last, .bfloat16);
            try growQuantBuf(s, &entry.values_biases, entry.initialized, entry.offset, new_cap, B, heads, sc_last, .bfloat16);
            entry.initialized = true;
        }

        // 5. slice_update each buffer at offset.
        try writeAtOffset(s, &entry.keys, entry.offset, new_kq.q);
        try writeAtOffset(s, &entry.values, entry.offset, new_vq.q);
        try writeAtOffset(s, &entry.keys_scales, entry.offset, new_kq.scales);
        try writeAtOffset(s, &entry.keys_biases, entry.offset, new_kq.biases);
        try writeAtOffset(s, &entry.values_scales, entry.offset, new_vq.scales);
        try writeAtOffset(s, &entry.values_biases, entry.offset, new_vq.biases);

        // 6. Update offset / step.
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 7. Build views for all 6 buffers.
        const total: c_int = @intCast(entry.offset);
        const is_decode = new_len == 1;
        const view_start: c_int = if (is_decode and max_seq > 0 and entry.offset > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;
        try buildSliceView(s, &entry.key_view, entry.keys, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values, total, view_start);
        try buildSliceView(s, &entry.key_scales_view, entry.keys_scales, total, view_start);
        try buildSliceView(s, &entry.key_biases_view, entry.keys_biases, total, view_start);
        try buildSliceView(s, &entry.value_scales_view, entry.values_scales, total, view_start);
        try buildSliceView(s, &entry.value_biases_view, entry.values_biases, total, view_start);

        // 8. Dequantize K/V for SDPA. Owner of these dense arrays is the
        //    DenseKVView returned to the caller.
        const dense_k = try kv_quant.dequantizeAffine(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, group_size, bits);
        errdefer _ = mlx.mlx_array_free(dense_k);
        const dense_v = try kv_quant.dequantizeAffine(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, group_size, bits);
        return .{ .k = dense_k, .v = dense_v, .owned = true };
    }

    fn updateDense(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];

        // 1. Free stale views — drops refcount on buffer → enables buffer donation
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);

        // 2. Get shape info from new_k: [B, heads, new_len, head_dim]
        const new_shape = mlx.getShape(new_k);
        const new_len: usize = @intCast(new_shape[2]);

        // 3. Grow buffer if needed
        if (!entry.initialized or entry.offset + new_len > bufferCapacity(entry.keys)) {
            const B = new_shape[0];
            const heads = new_shape[1];
            const head_dim = new_shape[3];
            const dtype = mlx.mlx_array_dtype(new_k);
            const needed = entry.offset + new_len;
            const n_chunks = (needed + chunk_step - 1) / chunk_step;
            const new_cap: c_int = @intCast(n_chunks * chunk_step);
            const buf_shape = [_]c_int{ B, heads, new_cap, head_dim };

            if (entry.initialized and entry.offset > 0) {
                // Growing existing buffer — create zeros and copy old data
                var new_k_buf = mlx.mlx_array_new();
                var new_v_buf = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&new_k_buf, &buf_shape, 4, dtype, s));
                try mlx.check(mlx.mlx_zeros(&new_v_buf, &buf_shape, 4, dtype, s));

                const off_c: c_int = @intCast(entry.offset);
                const su_start = [_]c_int{ 0, 0, 0, 0 };
                const su_stop = [_]c_int{ B, heads, off_c, head_dim };
                const su_strides = [_]c_int{ 1, 1, 1, 1 };

                var old_k_data = mlx.mlx_array_new();
                var old_v_data = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&old_k_data, entry.keys, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&old_v_data, entry.values, &su_start, 4, &su_stop, 4, &su_strides, 4, s));

                var updated_k = mlx.mlx_array_new();
                var updated_v = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&updated_k, new_k_buf, old_k_data, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
                try mlx.check(mlx.mlx_slice_update(&updated_v, new_v_buf, old_v_data, &su_start, 4, &su_stop, 4, &su_strides, 4, s));

                _ = mlx.mlx_array_free(old_k_data);
                _ = mlx.mlx_array_free(old_v_data);
                _ = mlx.mlx_array_free(new_k_buf);
                _ = mlx.mlx_array_free(new_v_buf);

                _ = mlx.mlx_array_free(entry.keys);
                _ = mlx.mlx_array_free(entry.values);
                entry.keys = updated_k;
                entry.values = updated_v;
            } else {
                // Fresh buffer — create zeros directly (no copy needed)
                _ = mlx.mlx_array_free(entry.keys);
                _ = mlx.mlx_array_free(entry.values);
                var new_k_buf = mlx.mlx_array_new();
                var new_v_buf = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&new_k_buf, &buf_shape, 4, dtype, s));
                try mlx.check(mlx.mlx_zeros(&new_v_buf, &buf_shape, 4, dtype, s));
                entry.keys = new_k_buf;
                entry.values = new_v_buf;
            }
            entry.initialized = true;
        }

        // 4. slice_update — write new_k/new_v into buffer at offset
        const buf_shape = mlx.getShape(entry.keys);
        const off: c_int = @intCast(entry.offset);
        const off_end: c_int = @intCast(entry.offset + new_len);
        const su_start = [_]c_int{ 0, 0, off, 0 };
        const su_stop = [_]c_int{ buf_shape[0], buf_shape[1], off_end, buf_shape[3] };
        const su_strides = [_]c_int{ 1, 1, 1, 1 };

        var updated_k = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&updated_k, entry.keys, new_k, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
        _ = mlx.mlx_array_free(entry.keys);
        entry.keys = updated_k;

        var updated_v = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&updated_v, entry.values, new_v, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
        _ = mlx.mlx_array_free(entry.values);
        entry.values = updated_v;

        // 5. Update offset and absolute step
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 6. Create views for attention.
        //    Skip slicing when the view covers the entire buffer — just reference it directly.
        //    This saves 2 C API calls per layer per token (84 calls/token for 42-layer models).
        const buf_cap = bufferCapacity(entry.keys);
        const total: c_int = @intCast(entry.offset);
        const is_decode = new_len == 1;
        const view_start: c_int = if (is_decode and max_seq > 0 and entry.offset > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;

        if (view_start == 0 and entry.offset == buf_cap) {
            // View covers the entire buffer — no slice needed (matches mlx-lm optimization)
            entry.key_view = mlx.mlx_array_new();
            entry.value_view = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&entry.key_view, entry.keys));
            try mlx.check(mlx.mlx_array_set(&entry.value_view, entry.values));
        } else {
            const cur_shape = mlx.getShape(entry.keys);
            const v_start = [_]c_int{ 0, 0, view_start, 0 };
            const v_stop = [_]c_int{ cur_shape[0], cur_shape[1], total, cur_shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            entry.key_view = mlx.mlx_array_new();
            entry.value_view = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&entry.key_view, entry.keys, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
            try mlx.check(mlx.mlx_slice(&entry.value_view, entry.values, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
        }

        return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
    }

    /// Read-side accessor: return a dense `[B,H,T,D]` K/V pair for the layer.
    /// In dense mode this aliases `key_view`/`value_view` (no-op deinit).
    /// In quant mode this dequantizes on the fly from the cache's stored
    /// triples (the returned arrays are owned and freed by `deinit`).
    /// SDPA call sites use this so they don't have to know the scheme.
    pub fn denseView(self: *KVCache, layer: u32, s: mlx.mlx_stream) !DenseKVView {
        const entry = &self.entries[layer];
        switch (self.config.scheme) {
            .off => return .{ .k = entry.key_view, .v = entry.value_view, .owned = false },
            .affine => {
                if (!entry.initialized) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                const dense_k = try kv_quant.dequantizeAffine(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, self.config.group_size, self.config.bits);
                errdefer _ = mlx.mlx_array_free(dense_k);
                const dense_v = try kv_quant.dequantizeAffine(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, self.config.group_size, self.config.bits);
                return .{
                    .k = dense_k,
                    .v = dense_v,
                    .owned = true,
                    // Borrow the cache's quant triples so fused-attn call
                    // sites can skip the dense materialization above (the
                    // dequant arrays still get computed — mlx is lazy, so
                    // the cost is only paid if SDPA actually reads them).
                    .k_triple_q = entry.key_view,
                    .k_triple_scales = entry.key_scales_view,
                    .k_triple_biases = entry.key_biases_view,
                    .v_triple_q = entry.value_view,
                    .v_triple_scales = entry.value_scales_view,
                    .v_triple_biases = entry.value_biases_view,
                    .has_quant_triple = true,
                    .bits = self.config.bits,
                    .group_size = self.config.group_size,
                };
            },
            .turboquant_2, .turboquant_4 => {
                if (!entry.initialized) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                const qs = if (self.quant_state) |*q| q else return error.MissingTurboState;
                // If we're reading before any write, the rotation matrices
                // aren't built yet — fall back to the raw view (which is
                // empty anyway when `initialized=false`, handled above).
                const li: usize = @intCast(layer);
                if (qs.rk_dim[li] == 0 or qs.rv_dim[li] == 0) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                const rk = qs.rk[li];
                const rv = qs.rv[li];
                const dense_k = try kv_quant.dequantizeTurbo(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, rk, self.config.group_size, self.config.bits);
                errdefer _ = mlx.mlx_array_free(dense_k);
                const dense_v = try kv_quant.dequantizeTurbo(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, rv, self.config.group_size, self.config.bits);
                return .{ .k = dense_k, .v = dense_v, .owned = true };
            },
        }
    }

    fn bufferCapacity(arr: mlx.mlx_array) usize {
        const shape = mlx.getShape(arr);
        if (shape.len < 3) return 0;
        return @intCast(shape[2]);
    }

    /// Affine-mode helpers: same buffer-grow / slice-update / view-build
    /// pattern as the dense path, parameterized over the buffer's last dim
    /// and dtype. Used six times per `updateAffine` (3 buffers × K and V).
    fn growQuantBuf(s: mlx.mlx_stream, buf: *mlx.mlx_array, initialized: bool, offset: usize, new_cap: c_int, B: c_int, heads: c_int, last_dim: c_int, dtype: mlx.mlx_dtype) !void {
        const buf_shape = [_]c_int{ B, heads, new_cap, last_dim };
        if (initialized and offset > 0) {
            var new_buf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            const off_c: c_int = @intCast(offset);
            const su_start = [_]c_int{ 0, 0, 0, 0 };
            const su_stop = [_]c_int{ B, heads, off_c, last_dim };
            const su_strides = [_]c_int{ 1, 1, 1, 1 };
            var old_data = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&old_data, buf.*, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            var updated = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&updated, new_buf, old_data, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            _ = mlx.mlx_array_free(old_data);
            _ = mlx.mlx_array_free(new_buf);
            _ = mlx.mlx_array_free(buf.*);
            buf.* = updated;
        } else {
            _ = mlx.mlx_array_free(buf.*);
            var new_buf = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            buf.* = new_buf;
        }
    }

    fn writeAtOffset(s: mlx.mlx_stream, buf: *mlx.mlx_array, offset: usize, new_chunk: mlx.mlx_array) !void {
        const new_shape = mlx.getShape(new_chunk);
        const new_len: c_int = new_shape[2];
        const buf_shape = mlx.getShape(buf.*);
        const off: c_int = @intCast(offset);
        const off_end: c_int = off + new_len;
        const su_start = [_]c_int{ 0, 0, off, 0 };
        const su_stop = [_]c_int{ buf_shape[0], buf_shape[1], off_end, buf_shape[3] };
        const su_strides = [_]c_int{ 1, 1, 1, 1 };
        var updated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, new_chunk, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    fn buildSliceView(s: mlx.mlx_stream, view: *mlx.mlx_array, buf: mlx.mlx_array, total: c_int, view_start: c_int) !void {
        const buf_cap = bufferCapacity(buf);
        if (view_start == 0 and @as(usize, @intCast(total)) == buf_cap) {
            try mlx.check(mlx.mlx_array_set(view, buf));
        } else {
            const cur_shape = mlx.getShape(buf);
            const v_start = [_]c_int{ 0, 0, view_start, 0 };
            const v_stop = [_]c_int{ cur_shape[0], cur_shape[1], total, cur_shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(view, buf, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
        }
    }

    pub fn seqLen(self: *const KVCache, layer: u32) usize {
        const entry = &self.entries[layer];
        if (!entry.initialized) return 0;
        return entry.offset;
    }

    /// Evaluate all KV cache entries to materialize them on GPU.
    /// Called after prefill to ensure the cache is in optimal memory layout for decode.
    pub fn evalState(self: *KVCache) void {
        // Collect all initialized entries into a vector and batch-eval them.
        // This matches mlx_lm's `mx.eval([c.state for c in cache])` pattern.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values);
            count += 1;
        }
        if (count > 0) {
            _ = mlx.mlx_eval(vec);
        }
    }

    /// Truncate the KV cache to keep only the first `len` tokens on the sequence axis.
    pub fn truncate(self: *KVCache, len: usize, s: mlx.mlx_stream) !void {
        self.step = len;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            if (len >= entry.offset) continue;

            // Free stale views (all 6: dense + 4 quant scale/bias views)
            _ = mlx.mlx_array_free(entry.key_view);
            _ = mlx.mlx_array_free(entry.value_view);
            entry.key_view = mlx.mlx_array_new();
            entry.value_view = mlx.mlx_array_new();
            if (self.config.scheme == .affine) {
                _ = mlx.mlx_array_free(entry.key_scales_view);
                _ = mlx.mlx_array_free(entry.key_biases_view);
                _ = mlx.mlx_array_free(entry.value_scales_view);
                _ = mlx.mlx_array_free(entry.value_biases_view);
                entry.key_scales_view = mlx.mlx_array_new();
                entry.key_biases_view = mlx.mlx_array_new();
                entry.value_scales_view = mlx.mlx_array_new();
                entry.value_biases_view = mlx.mlx_array_new();
            }

            if (len == 0) {
                _ = mlx.mlx_array_free(entry.keys);
                _ = mlx.mlx_array_free(entry.values);
                entry.keys = mlx.mlx_array_new();
                entry.values = mlx.mlx_array_new();
                if (self.config.scheme == .affine) {
                    _ = mlx.mlx_array_free(entry.keys_scales);
                    _ = mlx.mlx_array_free(entry.keys_biases);
                    _ = mlx.mlx_array_free(entry.values_scales);
                    _ = mlx.mlx_array_free(entry.values_biases);
                    entry.keys_scales = mlx.mlx_array_new();
                    entry.keys_biases = mlx.mlx_array_new();
                    entry.values_scales = mlx.mlx_array_new();
                    entry.values_biases = mlx.mlx_array_new();
                }
                entry.initialized = false;
                entry.offset = 0;
                continue;
            }

            // Just update offset — the buffer still holds data but views will
            // only expose [0:len]. No need to shrink the pre-allocated buffer.
            entry.offset = len;

            // Recreate views for the truncated range
            const shape = mlx.getShape(entry.keys);
            if (shape.len < 4) continue;
            const seq_end: c_int = @intCast(len);
            const v_start = [_]c_int{ 0, 0, 0, 0 };
            const v_stop = [_]c_int{ shape[0], shape[1], seq_end, shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&entry.key_view, entry.keys, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
            try mlx.check(mlx.mlx_slice(&entry.value_view, entry.values, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
            if (self.config.scheme == .affine) {
                const sc_shape = mlx.getShape(entry.keys_scales);
                const sv_stop = [_]c_int{ sc_shape[0], sc_shape[1], seq_end, sc_shape[3] };
                try mlx.check(mlx.mlx_slice(&entry.key_scales_view, entry.keys_scales, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.key_biases_view, entry.keys_biases, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.value_scales_view, entry.values_scales, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.value_biases_view, entry.values_biases, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
            }
        }
    }
};

/// Snapshot of a `KVCache` at a point in time. Owns its array handles (which
/// share buffers with the source via refcount) and frees them in `deinit`.
/// Created by `KVCache.snapshot()` and consumed by `KVCache.restore()`.
pub const KVCacheSnapshot = struct {
    entries: []KVCacheEntry,
    step: usize,
    allocator: std.mem.Allocator,
    config: KVQuantConfig,

    pub fn deinit(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
    }
};

// ── SSM Cache (for GatedDeltaNet linear attention layers) ──

pub const SSMCacheEntry = struct {
    conv_state: mlx.mlx_array, // [B, kernel-1, conv_dim]
    ssm_state: mlx.mlx_array, // [B, Hv, Dv, Dk]
    initialized: bool,
    /// PLD spec-decode capture: per-position SSM states recorded by the GDN
    /// verify forward (`Transformer.spec_capture_ssm`). Shape [T, B, Hv, Dv, Dk]
    /// where T = verify length. Lets partial-accept rollback pick the
    /// accepted-position state WITHOUT a re-forward — the verify already ran the
    /// (sequential, expensive on a 48-layer GatedDeltaNet trunk) recurrence, so
    /// re-running it for the accepted prefix is pure waste. Null outside a
    /// capturing verify; freed at the end of every `nextPld` round.
    spec_state_seq: mlx.mlx_array = .{ .ctx = null },
    /// PLD spec-decode capture: the conv1d input `[B, (kernel-1)+T, conv_dim]`
    /// of the verify forward, so the accepted-position `conv_state` is just a
    /// slice (no re-forward). Null outside capture.
    spec_conv_input: mlx.mlx_array = .{ .ctx = null },
};

/// SSM snapshot value. Holds clones of conv_state and ssm_state via refcount —
/// the underlying buffer is shared with the source entry, but the snapshot
/// owns its own array handles and frees them on deinit. Used for
/// speculative-decoding rollback (PLD) where we must be able to revert one
/// decode step on a hybrid model.
pub const SSMCacheEntrySnapshot = struct {
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    initialized: bool,
};

pub fn ssmSnapshot(src: *const SSMCacheEntry) SSMCacheEntrySnapshot {
    var out: SSMCacheEntrySnapshot = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = src.initialized,
    };
    // mlx_array_set increments refcount on the underlying buffer; both handles
    // point at the same data. Subsequent writes to src.conv_state/ssm_state
    // create NEW handles so the snapshot's view is unaffected.
    //
    // We must guard each field independently — the two states are populated by
    // DIFFERENT code paths and either may legitimately be null even when
    // `initialized == true`:
    //   - LFM2 `gatedConv` writes only `conv_state` (sets `initialized=true`),
    //     so `ssm_state.ctx == null` for the lifetime of that layer.
    //   - Mamba2/GatedDeltaNet flip the order: `conv1dWithCache` sets
    //     `initialized=true` BEFORE the recurrence body initializes
    //     `ssm_state`, so a snapshot taken in the middle would see a null
    //     ssm_state. (Currently snapshots are only taken between full forward
    //     passes, but defensive null-handling prevents future regressions.)
    //
    // Calling `mlx_array_set` with a null source aborts the process via mlx-c's
    // default error handler ("expected a non-empty mlx_array"), so we cannot
    // rely on `try mlx.check(...)`.
    if (src.conv_state.ctx != null) {
        _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
    }
    if (src.ssm_state.ctx != null) {
        _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
    }
    return out;
}

pub fn ssmSnapshotDeinit(snap: *SSMCacheEntrySnapshot) void {
    _ = mlx.mlx_array_free(snap.conv_state);
    _ = mlx.mlx_array_free(snap.ssm_state);
}

pub fn ssmRestore(dst: *SSMCacheEntry, snap: *const SSMCacheEntrySnapshot) !void {
    _ = mlx.mlx_array_free(dst.conv_state);
    _ = mlx.mlx_array_free(dst.ssm_state);
    dst.conv_state = mlx.mlx_array_new();
    dst.ssm_state = mlx.mlx_array_new();
    dst.initialized = snap.initialized;
    // Mirror snapshot's per-field null guard — the snapshot may legitimately
    // have a null ssm_state (LFM2 gated_conv layers) or null conv_state.
    if (snap.conv_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.conv_state, snap.conv_state));
    }
    if (snap.ssm_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&dst.ssm_state, snap.ssm_state));
    }
}

/// Free the transient PLD spec-decode capture buffers on an SSM entry
/// (`spec_state_seq` / `spec_conv_input`). Idempotent — safe to call on an
/// entry that never captured. Called at the end of every `nextPld` round and
/// in the cache teardown paths so the capture never outlives a round.
pub fn ssmFreeSpecCapture(entry: *SSMCacheEntry) void {
    if (entry.spec_state_seq.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_state_seq);
        entry.spec_state_seq = .{ .ctx = null };
    }
    if (entry.spec_conv_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_conv_input);
        entry.spec_conv_input = .{ .ctx = null };
    }
}

/// PLD partial-accept rollback for a GatedDeltaNet layer, using the
/// per-position capture from the verify forward instead of a re-forward.
/// Sets `conv_state` + `ssm_state` to the state AFTER processing the first
/// `1 + accepted` verify tokens (t1 + accepted accepted drafts).
///
/// Numerically identical to a fresh forward over `[t1, draft[0..accepted-1]]`:
/// the GDN recurrence is sequential, so the state captured at position
/// `accepted` of a length-(1+m) verify run is the exact same float32→bf16
/// value a length-(1+accepted) run would store as its final state. Likewise
/// `conv_state` is the same windowed slice of the same conv input. So this
/// preserves PLD's byte-equivalence guarantee (pinned by
/// `tests/test_pld_equivalence.sh`).
///
/// No-op when the entry holds no capture (non-GDN hybrid layer); the caller
/// only takes the fast path when capture succeeded.
pub fn ssmRollbackFromCapture(entry: *SSMCacheEntry, accepted: u32, s: mlx.mlx_stream) !void {
    if (entry.spec_state_seq.ctx == null) return;

    const seq_shape = mlx.getShape(entry.spec_state_seq); // [T, B, Hv, Dv, Dk]
    const acc: c_int = @intCast(accepted);

    // ssm_state = spec_state_seq[accepted]  →  [B, Hv, Dv, Dk]
    {
        const start = [_]c_int{ acc, 0, 0, 0, 0 };
        const stop = [_]c_int{ acc + 1, seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
        const strides = [_]c_int{ 1, 1, 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, entry.spec_state_seq, &start, 5, &stop, 5, &strides, 5, s));
        defer _ = mlx.mlx_array_free(sliced);
        const new_shape = [_]c_int{ seq_shape[1], seq_shape[2], seq_shape[3], seq_shape[4] };
        var reshaped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&reshaped, sliced, &new_shape, 4, s));
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = reshaped;
    }

    // conv_state = spec_conv_input[:, (1+accepted) : (1+accepted)+(kernel-1), :]
    if (entry.spec_conv_input.ctx != null) {
        const ci_shape = mlx.getShape(entry.spec_conv_input); // [B, (k-1)+T, conv_dim]
        const t_len = seq_shape[0]; // verify length T
        const km1 = ci_shape[1] - t_len; // kernel - 1
        const cstart: c_int = @intCast(1 + accepted);
        const start = [_]c_int{ 0, cstart, 0 };
        const stop = [_]c_int{ ci_shape[0], cstart + km1, ci_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv, entry.spec_conv_input, &start, 3, &stop, 3, &strides, 3, s));
        _ = mlx.mlx_array_free(entry.conv_state);
        entry.conv_state = new_conv;
    }
}

/// Phase 1 (performance-plan): per-position SSM checkpoint covering ALL hybrid
/// layers in the model. Captures the SSM/conv state after the model has been
/// forwarded over the first `pos` tokens of some prompt. Used by the hot prefix
/// cache to restore mid-sequence SSM state on a multi-turn warm request, so we
/// don't have to cold-prefill the shared prefix every turn.
///
/// A single checkpoint = one `SSMCacheEntrySnapshot` per layer (matching the
/// shape of `ctx.ssm_entries`). Layers whose ssm/conv state is uninitialized
/// at snapshot time hold a null-handle snapshot — `ssmRestore` is null-safe.
///
/// Memory: BF16 GatedDeltaNet state is ~260 KB per layer at typical
/// configurations. A 48-layer 1k-token snapshot stride 128 stores 8
/// checkpoints × 48 layers × 260 KB ≈ 100 MB. Bounded via
/// `HotPrefixCache.max_kv_bytes` (counts toward the same budget as KV).
pub const SSMCheckpoint = struct {
    /// 1-based KV position immediately AFTER forwarding `pos` tokens. The
    /// caller restores the slot's KVCache to exactly this position alongside
    /// the SSM state, so the model behaves as if it had only seen the first
    /// `pos` tokens of the prompt.
    pos: usize,
    /// Per-layer SSM snapshots — same length as `ctx.ssm_entries`. Non-SSM
    /// layers (plain attention) get a null-handle snapshot which is a no-op
    /// in `ssmRestore`.
    layers: []SSMCacheEntrySnapshot,

    pub fn deinit(self: *SSMCheckpoint, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| ssmSnapshotDeinit(l);
        allocator.free(self.layers);
        self.layers = &[_]SSMCacheEntrySnapshot{};
        self.pos = 0;
    }
};

/// Snapshot of every entry in `ssm_entries` at the current point. Caller owns
/// the resulting buffer (free via `SSMCheckpoint.deinit`).
///
/// Unlike the per-decode-step `ssmSnapshot` (PLD rollback — transient, so a
/// cheap refcount-share is right), checkpoints OUTLIVE the request inside the
/// hot prefix cache. The live conv/ssm states are routinely shared-buffer
/// SLICES of much larger parents (the prefill chunk's conv input
/// [B,(k-1)+T,C], the GDN capture sequence), so sharing the handle silently
/// retained the whole parent — the ~3.4x "[hot-cache] resident" under-count
/// on hybrid archs. `materializedOwnedCopy` forces a fresh buffer; the copies
/// are evaluated here, in one batch, so the parents' lifetimes end with the
/// chunk's activation graph instead of with the cache entry.
pub fn captureSsmCheckpoint(
    allocator: std.mem.Allocator,
    ssm_entries: []const SSMCacheEntry,
    pos: usize,
    s: mlx.mlx_stream,
) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        // Per-field null guards mirror `ssmSnapshot` — either state may
        // legitimately be null (LFM2 gated_conv never sets ssm_state).
        if (src.conv_state.ctx != null) {
            const c = try materializedOwnedCopy(s, src.conv_state);
            _ = mlx.mlx_array_free(out.conv_state);
            out.conv_state = c;
        }
        if (src.ssm_state.ctx != null) {
            const c = try materializedOwnedCopy(s, src.ssm_state);
            _ = mlx.mlx_array_free(out.ssm_state);
            out.ssm_state = c;
        }
        layers[i] = out;
        built = i + 1;
    }
    // Materialize all copies in one batch so the parent buffers can be
    // released with the prefill chunk's graph. Without this eval the lazy
    // copy node itself keeps the parent alive.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            if (l.conv_state.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, l.conv_state);
                count += 1;
            }
            if (l.ssm_state.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, l.ssm_state);
                count += 1;
            }
        }
        if (count > 0) _ = mlx.mlx_eval(vec);
    }
    return .{ .pos = pos, .layers = layers };
}

/// Force a REAL copy of `x` into a freshly allocated buffer. `mlx_copy` and
/// `mlx_contiguous` both take shared-buffer fast paths (Copy is a view op;
/// contiguous no-ops when the view's strides read as row-contiguous, which a
/// size-1-leading-dim slice does), so neither breaks the alias to a slice's
/// parent buffer. Adding a same-dtype scalar zero always runs a kernel with a
/// newly allocated output. Buffer donation can't alias either: the caller
/// still holds `x`, so its buffer is never donated.
pub fn materializedOwnedCopy(s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    const scalar_shape = [_]c_int{1};
    try mlx.check(mlx.mlx_zeros(&zero, &scalar_shape, 0, mlx.mlx_array_dtype(x), s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, x, zero, s));
    return out;
}

/// Restore every layer of `ssm_entries` from `cp`. Mirrors the per-layer
/// `ssmRestore` pattern — null-safe on either side.
pub fn restoreSsmCheckpoint(
    ssm_entries: []SSMCacheEntry,
    cp: *const SSMCheckpoint,
) !void {
    if (ssm_entries.len != cp.layers.len) return error.SsmCheckpointLayerMismatch;
    for (ssm_entries, cp.layers) |*dst, *src| {
        try ssmRestore(dst, src);
    }
}

/// Total bytes held by an SSM checkpoint (sum of conv_state + ssm_state across
/// all layers). Used for hot-cache memory budgeting alongside `KVCacheSnapshot`
/// bytes.
pub fn ssmCheckpointBytes(cp: *const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cp.layers) |l| {
        if (l.conv_state.ctx != null) {
            total += @as(u64, mlx.mlx_array_size(l.conv_state)) * @as(u64, mlx.mlx_array_itemsize(l.conv_state));
        }
        if (l.ssm_state.ctx != null) {
            total += @as(u64, mlx.mlx_array_size(l.ssm_state)) * @as(u64, mlx.mlx_array_itemsize(l.ssm_state));
        }
    }
    return total;
}

// ── Prompt Cache (snapshot of KV + SSM state for prefix reuse) ──

pub const PrefillCache = struct {
    tokens: []u32,
    kv_entries: []KVCacheEntry,
    offsets: []usize,
    kv_step: usize,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrefillCache) void {
        self.allocator.free(self.tokens);
        for (self.kv_entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.kv_entries);
        self.allocator.free(self.offsets);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            self.allocator.free(entries);
        }
    }
};

// ── Standard model per-layer weights ──

// ── BERT encoder-only layer weights ──

const BertLayerWeights = struct {
    // Self-attention (separate Q/K/V projections with real bias)
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    q_bias: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    k_bias: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    v_bias: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    o_bias: mlx.mlx_array,
    attn_norm_w: mlx.mlx_array,
    attn_norm_b: mlx.mlx_array,
    // MLP: intermediate -> GELU -> output
    inter_w: mlx.mlx_array,
    inter_s: mlx.mlx_array,
    inter_b: mlx.mlx_array,
    inter_bias: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
    out_bias: mlx.mlx_array,
    out_norm_w: mlx.mlx_array,
    out_norm_b: mlx.mlx_array,
};

// ── Standard decoder-only layer weights ──

const LayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    pre_ff_norm: ?mlx.mlx_array,
    post_ff_norm: ?mlx.mlx_array,
    q_norm: ?mlx.mlx_array,
    k_norm: ?mlx.mlx_array,
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    // Additive qkv-projection biases (Qwen2's `q/k/v_proj.bias`), distinct from
    // the quant `*_b` biases above. Empty-ctx when the arch has none (qwen3,
    // llama, mistral). Applied via `qmatmulMaybeBias` in the forward.
    q_bias: mlx.mlx_array = .{},
    k_bias: mlx.mlx_array = .{},
    v_bias: mlx.mlx_array = .{},
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    // Gemma 4: per-layer scalar, PLE weights
    layer_scalar: ?mlx.mlx_array = null,
    ple_gate_w: ?mlx.mlx_array = null,
    ple_gate_s: ?mlx.mlx_array = null,
    ple_gate_b: ?mlx.mlx_array = null,
    ple_proj_w: ?mlx.mlx_array = null,
    ple_proj_s: ?mlx.mlx_array = null,
    ple_proj_b: ?mlx.mlx_array = null,
    ple_norm: ?mlx.mlx_array = null,
    // KV sharing: source layer index (null = compute own KV)
    kv_source: ?u32 = null,
    // Gemma 4 (31B): V aliases K projection within this layer (no v_proj weight loaded)
    k_eq_v: bool = false,
};

// ── MoE model per-layer weights ──

const FullAttnWeights = struct {
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    // Laguna softplus per-head output gate (self_attn.g_proj, bf16, no
    // scales/biases). Null-ctx on every other arch → skipped by
    // appendFullAttnWeights and by lagunaAttnWith's gate branch.
    g_w: mlx.mlx_array = .{ .ctx = null },
    g_s: mlx.mlx_array = .{ .ctx = null },
    g_b: mlx.mlx_array = .{ .ctx = null },
};

const LinearAttnWeights = struct {
    // For separate projections (qwen3_5_moe): qkv=QKV, z=Z, a=A, b=B
    // For combined projections (qwen3_next): qkv=QKVZ, b=BA, z/a unused
    combined_proj: bool = false,
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
};

/// DiffusionGemma self-conditioning module: a GeGLU FFN over the previous
/// denoising step's soft embeddings, added to the canvas token embeddings
/// and re-normalized (scale-free) before decoder layer 0.
///   out = rms_norm_no_scale(embeds + down(geglu(gate(pre_norm(sig)), up(pre_norm(sig)))))
const SelfCondWeights = struct {
    pre_norm: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

const DenseMlpWeights = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

pub const MoeMlpWeights = struct {
    router_w: mlx.mlx_array,
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    switch_gate_w: mlx.mlx_array,
    switch_gate_s: mlx.mlx_array,
    switch_gate_b: mlx.mlx_array,
    switch_up_w: mlx.mlx_array,
    switch_up_s: mlx.mlx_array,
    switch_up_b: mlx.mlx_array,
    switch_down_w: mlx.mlx_array,
    switch_down_s: mlx.mlx_array,
    switch_down_b: mlx.mlx_array,
    shared_gate_w: mlx.mlx_array,
    shared_gate_s: mlx.mlx_array,
    shared_gate_b: mlx.mlx_array,
    shared_up_w: mlx.mlx_array,
    shared_up_s: mlx.mlx_array,
    shared_up_b: mlx.mlx_array,
    shared_down_w: mlx.mlx_array,
    shared_down_s: mlx.mlx_array,
    shared_down_b: mlx.mlx_array,
    // Shared expert gating (Qwen3.5; null for Gemma 4)
    shared_expert_gate_w: ?mlx.mlx_array = null,
    shared_expert_gate_s: ?mlx.mlx_array = null,
    shared_expert_gate_b: ?mlx.mlx_array = null,
    // Sigma-MoE routing (Gemma 4; null for Qwen3.5)
    router_scale: ?mlx.mlx_array = null,
    per_expert_scale: ?mlx.mlx_array = null,
    // Hy3 (hy_v3) sigmoid routing: f32 [num_experts] bias added to the sigmoid
    // scores for top-k SELECTION only (weights come from the unbiased scores).
    // Non-null expert_bias routes moeMLP2 through hy3RoutingChain.
    expert_bias: ?mlx.mlx_array = null,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
    // Hy3 shared expert is ALWAYS added, with no shared_expert_gate.
    shared_ungated: bool = false,
};

const HybridMlpWeights = union(enum) {
    dense: DenseMlpWeights,
    moe: MoeMlpWeights,
};

const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: HybridMlpWeights,
    // Gemma 4 MoE: separate shared expert MLP (null for Qwen3.5)
    shared_mlp: ?DenseMlpWeights = null,
    // Gemma 4 feedforward norms (null for Qwen3.5)
    pre_ff_norm: ?mlx.mlx_array = null,
    post_ff_norm: ?mlx.mlx_array = null,
    pre_ff_norm_2: ?mlx.mlx_array = null,
    post_ff_norm_1: ?mlx.mlx_array = null,
    post_ff_norm_2: ?mlx.mlx_array = null,
    layer_scalar: ?mlx.mlx_array = null,
    // DiffusionGemma: the causal ENCODER pass multiplies the layer output by
    // its own scalar (model.encoder.language_model.layers.N.layer_scalar) —
    // the only untied encoder text params. The bidirectional decoder pass
    // uses `layer_scalar` above. Null for every other arch.
    encoder_layer_scalar: ?mlx.mlx_array = null,
};

// ── Hybrid layer weights (LFM2, Nemotron-H) ──

const GatedConvWeights = struct {
    in_proj_w: mlx.mlx_array, // [3*hidden, hidden] → B, C, x split
    in_proj_s: mlx.mlx_array,
    in_proj_b: mlx.mlx_array,
    conv_w: mlx.mlx_array, // [hidden, kernel, 1] depthwise
    out_proj_w: mlx.mlx_array, // [hidden, hidden]
    out_proj_s: mlx.mlx_array,
    out_proj_b: mlx.mlx_array,
};

const Mamba2Weights = struct {
    in_proj_w: mlx.mlx_array,
    in_proj_s: mlx.mlx_array,
    in_proj_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array, // depthwise conv
    conv1d_b: ?mlx.mlx_array, // optional bias
    A_log: mlx.mlx_array, // static state matrix (log-space)
    D: mlx.mlx_array, // skip connection
    dt_bias: mlx.mlx_array, // time-step bias
    norm_w: mlx.mlx_array, // output normalization
    out_proj_w: mlx.mlx_array,
    out_proj_s: mlx.mlx_array,
    out_proj_b: mlx.mlx_array,
};

const SimpleMlpWeights = struct {
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

const HybridOp = union(enum) {
    gated_conv: GatedConvWeights,
    full_attn: FullAttnWeights,
    mamba2: Mamba2Weights,
    dense_mlp: DenseMlpWeights, // gated MLP (SwiGLU)
    simple_mlp: SimpleMlpWeights, // ungated MLP (ReLU^2)
};

const HybridLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_norm: ?mlx.mlx_array, // null for single-op blocks (Nemotron-H)
    op: HybridOp,
    mlp: ?DenseMlpWeights, // optional MLP after mixer (LFM2: always; Nemotron-H: null)
};

// ── Quantization params cache ──
// A tiny lock-free pointer → (bits, group_size) cache. Quantized weights are loaded
// once at init and reused for every forward pass; we detect (or pre-bind) params on
// first touch and serve hits thereafter. Uses open addressing with linear probing on
// a fixed-size array — this fits in L1 and keeps the cost of a lookup to ~5ns,
// matching the perf commit's intent of "eliminate per-call detect overhead" while
// supporting mixed-precision quantization (Gemma-4 MoE per-layer bits, etc.).
const BITS_CACHE_CAP: usize = 1024; // plenty for 60 layers × ~10 quant weights × factor
pub const QuantParams = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };

/// Pure inputs for the calibrated M5 MTP cost profile. The profile was
/// measured on a dense trunk whose actual lm_head takes the NAX lane at both
/// verify widths produced by depth 8 (M=8 and M=9); every field below mirrors
/// a production dispatch precondition rather than trusting model-wide config.
const MtpNaxProfileInputs = struct {
    dense_model: bool,
    calibrated_model: bool,
    profiled_affine_trunk: bool,
    model_quant: QuantParams,
    weight_present: bool,
    packed_weight: bool,
    scales_present: bool,
    biases_present: bool,
    quant: QuantParams,
    K: c_int,
    N: c_int,
    packed_k: c_int,
    verify_on: bool,
    lane_on: bool,
    available: bool,
    min_m: c_int,
};

fn mtpNaxCalibratedModelFrom(config: *const ModelConfig, lm_head_n: c_int) bool {
    return std.mem.eql(u8, config.model_type, "qwen3_5_moe") and
        !config.isMoe() and
        config.hidden_size == 5120 and
        config.intermediate_size == 17408 and
        config.num_hidden_layers == 64 and
        config.vocab_size == 248320 and
        lm_head_n == 248320 and
        config.num_attention_heads == 24 and
        config.num_key_value_heads == 4 and
        config.head_dim == 256 and
        config.full_attention_interval == 4 and
        config.linear_num_key_heads == 16 and
        config.linear_num_value_heads == 48 and
        config.linear_key_head_dim == 128 and
        config.linear_value_head_dim == 128 and
        config.attn_output_gate and
        !config.tie_word_embeddings;
}

fn mtpNaxProfileEnabledFrom(input: MtpNaxProfileInputs) bool {
    if (!input.dense_model or !input.calibrated_model or !input.profiled_affine_trunk) return false;
    if (!input.weight_present or !input.packed_weight) return false;
    if (!input.scales_present or !input.biases_present) return false;
    if (input.model_quant.bits != 4 or input.model_quant.mode != .affine) return false;
    if (input.model_quant.group_size != 64) return false;
    if (input.quant.bits != 4 or input.quant.mode != .affine) return false;
    if (input.quant.group_size != 64) return false;
    if (input.K <= 0 or input.N <= 0 or input.packed_k <= 0) return false;
    if (@mod(input.K, 8) != 0 or input.packed_k != @divExact(input.K, 8)) return false;

    return verifyQmmNaxEnabledForMFrom(
        8,
        input.K,
        input.N,
        input.verify_on,
        input.lane_on,
        input.available,
        input.min_m,
    ) and verifyQmmNaxEnabledForMFrom(
        9,
        input.K,
        input.N,
        input.verify_on,
        input.lane_on,
        input.available,
        input.min_m,
    );
}

fn mtpNaxAffineProjectionMatchesQuant(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    expected: QuantParams,
) bool {
    if (w.ctx == null or sc.ctx == null or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or
        mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    if (in_dim == 0 or out_dim == 0 or
        in_dim > std.math.maxInt(c_int) or out_dim > std.math.maxInt(c_int)) return false;
    const out: c_int = @intCast(out_dim);
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const b_shape = mlx.getShape(bi);
    if (w_shape.len != 2 or s_shape.len != 2 or b_shape.len != 2) return false;
    if (w_shape[0] != out or s_shape[0] != out or b_shape[0] != out) return false;
    if (s_shape[1] != b_shape[1]) return false;
    const qp = affineParamsFromGeometry(w, sc, in_dim) orelse return false;
    if (qp.bits != expected.bits or
        qp.group_size != expected.group_size or
        qp.mode != expected.mode) return false;

    // Every material projection on the calibrated surface must take NAX at
    // both depth-8 verify widths. Tiny per-head A/B gates and the measured
    // narrow q5/q6 regressions remain on stock by design.
    if (out_dim >= 512 and mixedNaxShapeEnabled(expected.bits, expected.group_size, @intCast(out_dim))) {
        const K: c_int = @intCast(in_dim);
        const N: c_int = @intCast(out_dim);
        if (!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, true, 8) or
            !verifyQmmNaxEnabledForMFrom(9, K, N, true, true, true, 8)) return false;
    }
    return true;
}

fn mtpNaxAffineProjectionMatches(
    config: *const ModelConfig,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = config.quant_bits,
        .group_size = config.quant_group_size,
        .mode = config.quant_mode,
    });
}

fn mtpNaxDenseMlpMatches(config: *const ModelConfig, mlp: *const DenseMlpWeights) bool {
    return mtpNaxAffineProjectionMatches(config, mlp.gate_w, mlp.gate_s, mlp.gate_b, config.hidden_size, config.intermediate_size) and
        mtpNaxAffineProjectionMatches(config, mlp.up_w, mlp.up_s, mlp.up_b, config.hidden_size, config.intermediate_size) and
        mtpNaxAffineProjectionMatches(config, mlp.down_w, mlp.down_s, mlp.down_b, config.intermediate_size, config.hidden_size);
}

const OQE_MLP_DOWN_BITS = "5555555555555555555555555556555644444444444444444444444444444444";
const OQE_LINEAR_Z_BITS = "5554444444545554555454444444444444444444444444444444444444444444";
const OQE_LINEAR_AB_BITS = "5554544444545554555455545554555444444444444444444444444444444444";
const OQE_LINEAR_OUT_BITS = "5554555455545554555455545554555455545554555455545554555455545554";
const OQE_FULL_QK_BITS = "4444444444444444444444444445444544444444444444444444444444444444";
const OQE_FULL_V_BITS = "4444444444444444444444444446444444444444444444444444444444444444";
const OQE_FULL_O_BITS = "4444444444454445444544444444444444444444444444444444444444444444";

fn oqeLayerBits(pattern: *const [64:0]u8, layer: usize) u32 {
    std.debug.assert(layer < 64);
    return pattern[layer] - '0';
}

fn mtpNaxOqeProjectionMatches(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    bits: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = bits,
        .group_size = 64,
        .mode = .affine,
    });
}

/// Exact resident-tensor fingerprint for Jundot's Qwen3.6-27B-oQ4e trunk.
/// The checkpoint is globally affine-4/gs64 with a fixed layer/role map of
/// affine q5/q6 overrides. Matching the full map avoids granting cap 8 to
/// unrelated "mixed 4-bit" checkpoints with a different round-cost surface.
fn mtpNaxOqeAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    if (config.isMoe() or config.full_attention_interval == 0) return false;
    if (config.num_hidden_layers != 64 or config.hidden_size == 0 or config.intermediate_size == 0) return false;
    const layers = maybe_layers orelse return false;
    if (layers.len != 64) return false;

    const full_out_wide = @as(u64, config.num_attention_heads) * config.head_dim;
    const full_q_out_wide = full_out_wide * 2;
    const kv_out_wide = @as(u64, config.num_key_value_heads) * config.head_dim;
    const linear_key_out_wide = @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim;
    const linear_out_wide = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const linear_qkv_out_wide = 2 * linear_key_out_wide + linear_out_wide;
    if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
        full_q_out_wide > std.math.maxInt(u32) or
        kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32) or
        linear_out_wide == 0 or linear_out_wide > std.math.maxInt(u32) or
        linear_qkv_out_wide > std.math.maxInt(u32)) return false;
    const full_out: u32 = @intCast(full_out_wide);
    const full_q_out: u32 = @intCast(full_q_out_wide);
    const kv_out: u32 = @intCast(kv_out_wide);
    const linear_out: u32 = @intCast(linear_out_wide);
    const linear_qkv_out: u32 = @intCast(linear_qkv_out_wide);

    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != config.isLinearLayer(@intCast(i))) return false;
        switch (layer.attn) {
            .full => |*attn| {
                if (!mtpNaxOqeProjectionMatches(attn.q_w, attn.q_s, attn.q_b, config.hidden_size, full_q_out, oqeLayerBits(OQE_FULL_QK_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.k_w, attn.k_s, attn.k_b, config.hidden_size, kv_out, oqeLayerBits(OQE_FULL_QK_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.v_w, attn.v_s, attn.v_b, config.hidden_size, kv_out, oqeLayerBits(OQE_FULL_V_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.o_w, attn.o_s, attn.o_b, full_out, config.hidden_size, oqeLayerBits(OQE_FULL_O_BITS, i))) return false;
            },
            .linear => |*attn| {
                if (attn.combined_proj) return false;
                if (!mtpNaxOqeProjectionMatches(attn.qkv_w, attn.qkv_s, attn.qkv_b, config.hidden_size, linear_qkv_out, 4) or
                    !mtpNaxOqeProjectionMatches(attn.z_w, attn.z_s, attn.z_b, config.hidden_size, linear_out, oqeLayerBits(OQE_LINEAR_Z_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.a_w, attn.a_s, attn.a_b, config.hidden_size, config.linear_num_value_heads, oqeLayerBits(OQE_LINEAR_AB_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.b_w, attn.b_s, attn.b_b, config.hidden_size, config.linear_num_value_heads, oqeLayerBits(OQE_LINEAR_AB_BITS, i)) or
                    !mtpNaxOqeProjectionMatches(attn.out_w, attn.out_s, attn.out_b, linear_out, config.hidden_size, oqeLayerBits(OQE_LINEAR_OUT_BITS, i))) return false;
            },
        }
        switch (layer.mlp) {
            .dense => |*mlp| {
                if (!mtpNaxOqeProjectionMatches(mlp.gate_w, mlp.gate_s, mlp.gate_b, config.hidden_size, config.intermediate_size, 4) or
                    !mtpNaxOqeProjectionMatches(mlp.up_w, mlp.up_s, mlp.up_b, config.hidden_size, config.intermediate_size, 4) or
                    !mtpNaxOqeProjectionMatches(mlp.down_w, mlp.down_s, mlp.down_b, config.intermediate_size, config.hidden_size, oqeLayerBits(OQE_MLP_DOWN_BITS, i))) return false;
            },
            .moe => return false,
        }
    }
    return true;
}

/// The depth-8 controller constants describe the measured homogeneous affine
/// Qwen3.6-27B trunk, not merely its architecture. Same-shape mixed checkpoints
/// (notably Unsloth Dynamic) may store individual projections as BF16 or mxfp8
/// while retaining a model-wide affine-4 config. Inspect every resident linear
/// used by the dense trunk so those checkpoints stay on the conservative cap 6.
fn mtpNaxUniformAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    if (config.isMoe() or config.full_attention_interval == 0) return false;
    if (config.hidden_size == 0 or config.intermediate_size == 0) return false;
    const layers = maybe_layers orelse return false;
    if (layers.len != @as(usize, @intCast(config.num_hidden_layers))) return false;

    const full_out_wide = @as(u64, config.num_attention_heads) * config.head_dim;
    const full_q_out_wide = full_out_wide * 2;
    const kv_out_wide = @as(u64, config.num_key_value_heads) * config.head_dim;
    const linear_key_out_wide = @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim;
    const linear_out_wide = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const linear_qkv_out_wide = 2 * linear_key_out_wide + linear_out_wide;
    if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
        full_q_out_wide > std.math.maxInt(u32) or
        kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32) or
        linear_key_out_wide == 0 or linear_key_out_wide > std.math.maxInt(u32) or
        linear_out_wide == 0 or linear_out_wide > std.math.maxInt(u32) or
        linear_qkv_out_wide > std.math.maxInt(u32)) return false;
    const full_out: u32 = @intCast(full_out_wide);
    const full_q_out: u32 = @intCast(full_q_out_wide);
    const kv_out: u32 = @intCast(kv_out_wide);
    const linear_out: u32 = @intCast(linear_out_wide);
    const linear_qkv_out: u32 = @intCast(linear_qkv_out_wide);

    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != config.isLinearLayer(@intCast(i))) return false;
        switch (layer.attn) {
            .full => |*attn| {
                if (!mtpNaxAffineProjectionMatches(config, attn.q_w, attn.q_s, attn.q_b, config.hidden_size, full_q_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.k_w, attn.k_s, attn.k_b, config.hidden_size, kv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.v_w, attn.v_s, attn.v_b, config.hidden_size, kv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.o_w, attn.o_s, attn.o_b, full_out, config.hidden_size)) return false;
            },
            .linear => |*attn| {
                if (attn.combined_proj) return false;
                if (!mtpNaxAffineProjectionMatches(config, attn.qkv_w, attn.qkv_s, attn.qkv_b, config.hidden_size, linear_qkv_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.z_w, attn.z_s, attn.z_b, config.hidden_size, linear_out) or
                    !mtpNaxAffineProjectionMatches(config, attn.a_w, attn.a_s, attn.a_b, config.hidden_size, config.linear_num_value_heads) or
                    !mtpNaxAffineProjectionMatches(config, attn.b_w, attn.b_s, attn.b_b, config.hidden_size, config.linear_num_value_heads) or
                    !mtpNaxAffineProjectionMatches(config, attn.out_w, attn.out_s, attn.out_b, linear_out, config.hidden_size)) return false;
            },
        }
        switch (layer.mlp) {
            .dense => |*mlp| if (!mtpNaxDenseMlpMatches(config, mlp)) return false,
            .moe => return false,
        }
    }
    return true;
}

const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = @splat(null),
    vals_bits: [BITS_CACHE_CAP]u8 = @splat(0),
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = @splat(0),
    vals_mode: [BITS_CACHE_CAP]u8 = @splat(0),

    inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert params for `key`. Returns false if the probe window saturated
    /// (caller should treat as "fall through to detection" — but realistically
    /// never fires given BITS_CACHE_CAP is 1024 vs ~2k weights max).
    fn put(self: *QuantParamsCache, key: *anyopaque, qp: QuantParams) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(qp.bits);
                self.vals_gs_div8[idx] = @intCast(qp.group_size / 8);
                self.vals_mode[idx] = @intFromEnum(qp.mode);
                return true;
            }
        }
        return false;
    }
};

// Backwards-compatibility alias — keeps existing field names readable.
const BitsCache = QuantParamsCache;

// ── Forward context ──
//
// Routes per-request mutable state (KV cache, MoE seq offset, SSM entries,
// hidden-state capture target, vision embeddings) into the forward pass via
// a single struct. The legacy single-slot path uses `Transformer.defaultCtx()`
// which points at the Transformer's own fields — semantically identical to the
// pre-refactor code. Concurrent batching (Phase 1+) constructs one ctx per
// in-flight request so multiple slots can share a Transformer's weights while
// owning their own caches.
pub const ForwardCtx = struct {
    cache: *KVCache,
    moe_seq_offset: *usize,
    ssm_entries: ?[]SSMCacheEntry,
    capture_hidden: ?*mlx.mlx_array,
    /// Like `capture_hidden` but receives the FULL post-final-norm hidden
    /// `[B, L, H]` (all positions, refcount-shared) instead of the last
    /// position only. Used by the Qwen MTP head, whose committed-history
    /// cache needs the trunk hidden at every verify/prefill position.
    capture_hidden_all: ?*mlx.mlx_array = null,
    /// PLD spec-decode: when true, the GatedDeltaNet trunk records per-position
    /// SSM/conv state during the verify forward (see `SSMCacheEntry.spec_*`),
    /// so partial-accept rollback needs no re-forward. Set only by `nextPld`
    /// for the verify pass on a GDN model; the flag reaches the layers via
    /// `Transformer.spec_capture_ssm`.
    capture_ssm_seq: bool = false,
    vision_embeddings: ?mlx.mlx_array,
    /// Qwen3-VL interleaved M-RoPE. `mrope_pos` is the server-computed flat
    /// [3 × mrope_total] i32 position-id table (axis-major: t,h,w rows), borrowed
    /// from the slot. `mrope_delta` shifts the scalar decode RoPE (decode tokens
    /// are text → t=h=w). `mrope_cos/sin_cur` are the per-prefill-chunk cos/sin
    /// tables, built once in forwardMoeWith and consumed by every full-attn layer
    /// that chunk, then freed. All null/0 for non-Qwen / text-only requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    /// Phase 2 (Plan ricky): when true, attention call sites consume the
    /// cache's quantized K/V triples directly via `kv_quant.quantAttention`
    /// instead of dequantizing through `DenseKVView`. Only effective when
    /// the cache scheme is .affine — TurboQuant + .off ignore this flag
    /// (TurboQuant needs the rotation undo step, which the fused path
    /// doesn't yet implement; .off has no quant triple to consume).
    /// Default false → unchanged dense SDPA path.
    kv_attn_fused: bool = false,
    /// Batched-embeddings: additive key-padding mask [B, 1, 1, T] consumed
    /// by the BERT encoder forward so padded positions never attend. Null
    /// (the default) keeps the unmasked single-sequence path.
    key_pad_mask: ?mlx.mlx_array = null,
    /// DiffusionGemma encoder pass: multiply each layer's output by the
    /// ENCODER layer scalar instead of the decoder's. Set by the diffusion
    /// runner for prompt prefill and committed-canvas re-encode passes.
    use_encoder_scalars: bool = false,
    /// DiffusionGemma encoder pass: the encoder only exists to fill the KV
    /// cache — skip the 262K-vocab lm_head projection and return the
    /// post-final-norm hidden instead of logits (caller frees either way).
    skip_lm_head: bool = false,
};

// ── Transformer ──

pub const Transformer = struct {
    config: ModelConfig,
    cache: KVCache,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array,
    final_norm: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    layers: []LayerWeights,

    // EmbeddingGemma sentence-transformers projection head (dense.0 →
    // dense.1, identity activations, no layer bias), applied between
    // mean-pool and L2-normalize. Borrowed from the weights map (map outlives
    // the Transformer; never freed here). Null-ctx when absent.
    dense0_w: mlx.mlx_array = .{},
    dense0_s: mlx.mlx_array = .{},
    dense0_b: mlx.mlx_array = .{},
    dense1_w: mlx.mlx_array = .{},
    dense1_s: mlx.mlx_array = .{},
    dense1_b: mlx.mlx_array = .{},

    owns_lm_head: bool,
    owns_norms: bool,
    embedding_mode: bool = false,

    gelu_coeff: ?mlx.mlx_array,
    gelu_inner: ?mlx.mlx_array,
    half: mlx.mlx_array,
    one: mlx.mlx_array,
    three: ?mlx.mlx_array,
    neg_one: ?mlx.mlx_array,

    // Gemma 4 PLE (Per-Layer Embeddings) global weights
    ple_emb_w: mlx.mlx_array, // embed_tokens_per_layer
    ple_emb_s: mlx.mlx_array,
    ple_emb_b: mlx.mlx_array,
    ple_proj_w: mlx.mlx_array, // per_layer_model_projection
    ple_proj_s: mlx.mlx_array,
    ple_proj_b: mlx.mlx_array,
    ple_proj_norm: mlx.mlx_array, // per_layer_projection_norm
    ple_proj_quantized: bool, // whether per_layer_model_projection is quantized
    // Gemma 4: logit softcapping scalar and v_norm weight (ones)
    softcap_scalar: ?mlx.mlx_array,
    v_norm_weight: ?mlx.mlx_array, // ones(head_dim) for param-free RMS norm
    v_norm_weight_global: ?mlx.mlx_array, // ones(global_head_dim)

    // DiffusionGemma: self-conditioning module weights (decoder-only) and a
    // ones(hidden_size) vector for its scale-free post_norm + the router-style
    // param-free norms in the diffusion decoder path. Null for other archs.
    self_cond: ?SelfCondWeights = null,
    ones_hidden: ?mlx.mlx_array = null,

    // Proportional RoPE frequencies for global/full attention layers (Gemma 4)
    rope_freqs_global: ?mlx.mlx_array,

    // Laguna YaRN (full-attention layers only): the per-dim RoPE denominator
    // array handed to mlx_fast_rope (rope_freqs_yarn, [rotary_dim/2] f32) and
    // the mscale vector (yarn_mscale, [head_dim] f32 = attention_factor on the
    // rotated dims, 1.0 on the pass-through tail). Null on every other arch.
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,

    // BERT encoder-only (null for decoder models)
    bert_layers: ?[]BertLayerWeights,
    bert_pos_w: mlx.mlx_array,
    bert_pos_s: mlx.mlx_array,
    bert_pos_b: mlx.mlx_array,
    bert_toktype_w: mlx.mlx_array,
    bert_toktype_s: mlx.mlx_array,
    bert_toktype_b: mlx.mlx_array,
    bert_emb_norm_w: mlx.mlx_array,
    bert_emb_norm_b: mlx.mlx_array,

    // MoE-specific (null/empty for standard models)
    moe_layers: ?[]MoeLayerWeights,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    // Pre-transposed plain-bf16 linear_attn weights owned by the Transformer
    // (Unsloth Dynamic checkpoints — null for the common all-quantized case).
    moe_owned_bf16: ?[]mlx.mlx_array = null,

    // When non-null, the next forward pass captures the post-final-norm
    // hidden state at the last position into the pointed-to array
    // (refcount-shared with the live forward graph). Set/cleared by
    // `forwardCaptureHidden`. Used by PLD verify-fusion and the Gemma 4
    // assistant drafter for h_prev seed.
    // Single-threaded: generation runs on one thread per Transformer.
    capture_hidden: ?*mlx.mlx_array = null,

    // Hybrid layers (LFM2, Nemotron-H)
    hybrid_layers: ?[]HybridLayerWeights,
    embedding_norm: ?mlx.mlx_array, // LFM2: RMS norm on embeddings

    // Prompt cache for prefix reuse across requests
    prompt_cache: ?PrefillCache,

    // Compiled forward pass closure (JIT-compiled for Metal kernel fusion)
    compiled_forward: ?mlx.mlx_closure = null,

    // Vision: set before prefill when images are present, cleared after.
    // Shape: [B, num_image_tokens, hidden_size]. Spliced at image_token_id positions.
    vision_embeddings: ?mlx.mlx_array = null,

    // Compiled closures (fuse ops into single kernels, matching mlx-lm's @mx.compile)
    compiled_gelu: ?mlx.mlx_closure = null,
    compiled_geglu: ?mlx.mlx_closure = null, // gelu(gate) * up → 1 kernel
    compiled_softcap: ?mlx.mlx_closure = null, // tanh(x/cap) * cap → 1 kernel
    compiled_moe_routing: ?mlx.mlx_closure = null, // negate→argpartition→slice→softmax→take→sum→expand→divide → 1 kernel
    compiled_hy3_routing: ?mlx.mlx_closure = null, // hy_v3 sigmoid+bias variant (2 inputs: logits, expert_bias)
    compiled_gdn_gate: ?mlx.mlx_closure = null, // exp(-exp(A_log)·softplus(a+dt_bias)) → 1 kernel (mirrors mlx-lm compute_g)

    // GatedDeltaNet per-token decode used to rebuild these on EVERY layer/step
    // (measured dispatch overhead on Qwen 3.6 hybrid). Built lazily on first
    // use, freed in deinit.
    gdn_ones_w: ?mlx.mlx_array = null, // ones([dk]) for parameter-free rms_norm
    gdn_q_scale: ?mlx.mlx_array = null, // bf16 scalar 1/dk
    gdn_k_scale: ?mlx.mlx_array = null, // bf16 scalar 1/sqrt(dk)
    /// PLD spec-decode: mirrors `ForwardCtx.capture_ssm_seq` for the current
    /// forward so `gatedDeltaNet`/`conv1dWithCache` (which don't take the ctx)
    /// can capture per-position state. Set+reset only inside `forwardMoeWith`.
    spec_capture_ssm: bool = false,

    // Per-weight quantization bit cache (see bitsFor). Populated lazily on first use.
    // Keyed by the scales array's ctx pointer (stable for the lifetime of a weight).
    // Used instead of config.quant_bits so mixed-precision models (Gemma-4 MoE, etc.)
    // with per-layer overrides work correctly while keeping zero per-call FFI overhead
    // after the first touch.
    bits_cache: BitsCache = .{},

    // True only for the exact measured Qwen3.6-27B architecture when its token
    // embedding and every resident trunk projection are homogeneous
    // affine-4/gs-64. Computed once at load; mixed checkpoints keep this false.
    mtp_uniform_affine_trunk: bool = false,
    // Exact oQ4e mixed q4/q5/q6 layer-role fingerprint. Kept separate from
    // the homogeneous profile because its measured NAX cost surface differs.
    mtp_oqe_affine_trunk: bool = false,

    pub fn init(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Transformer {
        // Use the current thread's default GPU stream rather than a dedicated stream.
        // mlx 0.31.2 made streams thread-local — a stream created on one thread isn't
        // visible to other threads, so a long-lived dedicated stream stored on Transformer
        // would break as soon as a different thread (e.g. an HTTP connection handler)
        // tried to use it. We re-bind `self.s` to the connection thread's default stream
        // via `useCurrentThreadStream` before each request.
        const s = mlx.gpuStream();
        const prefix = config.weight_prefix;

        var name_buf: [256]u8 = undefined;

        // BERT encoders get their own weight layout + forward; bidirectional
        // decoder archs (EmbeddingGemma) load through the standard arm and
        // dispatch to forwardGemma3EncoderWith.
        if (config.is_encoder_only and !config.use_bidirectional_attention) return initBert(io, allocator, config, weights, &name_buf, s);
        if (std.mem.eql(u8, config.model_type, "deepseek_v4")) {
            log.err("MLX-format deepseek_v4 is not supported — load the GGUF checkpoint via the ds4 engine instead\n", .{});
            return error.UnsupportedModelType;
        }

        // Embeddings: Nemotron-H uses "backbone.embeddings", others use "{prefix}.embed_tokens"
        const is_nemotron = std.mem.eql(u8, config.model_type, "nemotron_h");
        const emb_w = if (is_nemotron)
            getWeightFmt(weights, &name_buf, "{s}.embeddings.weight", prefix)
        else
            getWeightFmt(weights, &name_buf, "{s}.embed_tokens.weight", prefix);
        // Dense bf16 (quant_bits==0): no scales/biases exist → null-ctx arrays
        // signal "plain bf16" to embedding()/dequantTake().
        // Bias-less quant modes (nvfp4/mxfp4/mxfp8): scales exist, biases are
        // OPTIONAL — absent on fp8 tensors, present on affine-override tensors
        // in mixed QAT checkpoints. A mandatory fetch would be a spurious
        // MISSING WEIGHT crash (issue #24).
        // Mixed checkpoints can also leave the WHOLE embedding table dense
        // beside quantized layers (hy_v3 2-bit ships a bf16 embed_tokens) —
        // scales-presence is decided by the TABLE's dtype (float ⇒ dense),
        // never by the config's global bits. A packed (uint32) table missing
        // its scales still crashes honestly.
        const bias_mandatory = config.quant_bits > 0 and config.quant_mode.hasBiases();
        const emb_dense = floatDtypeTable(mlx.mlx_array_dtype(emb_w));
        const emb_s_arr = if (config.quant_bits == 0 or emb_dense)
            mlx.mlx_array_new()
        else if (is_nemotron)
            getWeightFmt(weights, &name_buf, "{s}.embeddings.scales", prefix)
        else
            getWeightFmt(weights, &name_buf, "{s}.embed_tokens.scales", prefix);
        const emb_b_arr = if (config.quant_bits == 0 or emb_dense)
            mlx.mlx_array_new()
        else if (is_nemotron)
            (if (bias_mandatory) getWeightFmt(weights, &name_buf, "{s}.embeddings.biases", prefix) else getWeightFmtOpt(weights, &name_buf, "{s}.embeddings.biases", prefix) orelse mlx.mlx_array_new())
        else
            (if (bias_mandatory) getWeightFmt(weights, &name_buf, "{s}.embed_tokens.biases", prefix) else getWeightFmtOpt(weights, &name_buf, "{s}.embed_tokens.biases", prefix) orelse mlx.mlx_array_new());

        const emb_scale: ?mlx.mlx_array = if (config.scale_embeddings)
            bf16Scalar(@sqrt(@as(f32, @floatFromInt(config.hidden_size))), s)
        else
            null;

        // Final norm: LFM2 uses "embedding_norm", Nemotron-H uses "norm_f", others use "norm"
        const is_lfm2 = std.mem.eql(u8, config.model_type, "lfm2");
        var final_norm: mlx.mlx_array = undefined;
        if (!config.has_final_norm) {
            final_norm = mlx.mlx_array_new(); // placeholder, unused
        } else if (is_lfm2) {
            final_norm = getWeightFmt(weights, &name_buf, "{s}.embedding_norm.weight", prefix);
        } else if (is_nemotron) {
            final_norm = getWeightFmt(weights, &name_buf, "{s}.norm_f.weight", prefix);
        } else {
            const final_norm_raw = getWeightFmt(weights, &name_buf, "{s}.norm.weight", prefix);
            final_norm = if (config.norm_has_offset) try addOne(final_norm_raw, s) else final_norm_raw;
            if (config.norm_has_offset) try mlx.check(mlx.mlx_array_eval(final_norm));
        }

        var lm_head_w: mlx.mlx_array = undefined;
        var lm_head_s: mlx.mlx_array = undefined;
        var lm_head_b: mlx.mlx_array = undefined;
        var owns_lm_head = false;

        {
            // lm_head prefix: "language_model.model" -> "language_model", "model" -> try root, else -> prefix
            const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
            if (maybe_lm_w) |w| {
                lm_head_w = w;
                // Dense bf16: no scales/biases → null-ctx; lmHeadProject() then
                // projects via a transposed view of the [vocab, hidden] weight.
                // Per-TENSOR dense detection (float dtype), same as embed_tokens
                // above — mixed checkpoints may quantize layers but not the head.
                const head_dense = config.quant_bits == 0 or floatDtypeTable(mlx.mlx_array_dtype(w));
                lm_head_s = if (head_dense) mlx.mlx_array_new() else getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
                lm_head_b = if (head_dense)
                    mlx.mlx_array_new()
                else if (bias_mandatory)
                    getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix)
                else
                    getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array_new();
                owns_lm_head = !config.tie_word_embeddings;
            } else if (weights.get("lm_head.weight")) |w| {
                lm_head_w = w;
                lm_head_s = weights.get("lm_head.scales") orelse emb_s_arr;
                lm_head_b = weights.get("lm_head.biases") orelse emb_b_arr;
                owns_lm_head = !config.tie_word_embeddings;
            } else if (config.tie_word_embeddings) {
                lm_head_w = emb_w;
                lm_head_s = emb_s_arr;
                lm_head_b = emb_b_arr;
            } else {
                log.err("MISSING WEIGHT: lm_head.weight\n", .{});
                unreachable;
            }
        }

        // EmbeddingGemma sentence-transformers projection head: folded into
        // the main safetensors as root-level dense.0/dense.1 by the
        // mlx-community conversion. Optional — absent on every other model.
        const dense0_w: mlx.mlx_array = weights.get("dense.0.weight") orelse .{};
        const dense0_s: mlx.mlx_array = weights.get("dense.0.scales") orelse .{};
        const dense0_b: mlx.mlx_array = weights.get("dense.0.biases") orelse .{};
        const dense1_w: mlx.mlx_array = weights.get("dense.1.weight") orelse .{};
        const dense1_s: mlx.mlx_array = weights.get("dense.1.scales") orelse .{};
        const dense1_b: mlx.mlx_array = weights.get("dense.1.biases") orelse .{};

        // Cache for KV (standard models use all entries, MoE only uses full-attn layers)
        const cache = try KVCache.init(allocator, config.num_hidden_layers);

        const need_gelu = config.hidden_act == .gelu_approx;
        const need_silu = config.hidden_act == .silu;

        // Load architecture-specific layer weights
        var layers: []LayerWeights = &.{};
        var moe_layers: ?[]MoeLayerWeights = null;
        var ssm_entries: ?[]SSMCacheEntry = null;
        var hybrid_layers: ?[]HybridLayerWeights = null;
        var moe_owned_bf16: ?[]mlx.mlx_array = null;

        if (config.has_hybrid_layers) {
            const hl = try initHybridLayers(allocator, config, weights, &name_buf, s);
            hybrid_layers = hl.hybrid_layers;
            ssm_entries = hl.ssm_entries;
        } else if (config.isMoe() or config.full_attention_interval > 0) {
            const ml = try initMoeLayers(allocator, config, weights, &name_buf, s);
            moe_layers = ml.moe_layers;
            moe_owned_bf16 = ml.owned_bf16;
            // ssm_entries are only meaningful when the family actually has
            // linear-attention (GDN) layers — full_attention_interval > 0.
            // A pure-attention MoE (qwen3_moe, Gemma 4 MoE) carrying non-null
            // ssm_entries makes the hot prefix cache classify the model as
            // hybrid and force a cold prefill on EVERY request: no checkpoint
            // can ever exist, so every lookup is a "hybrid miss" (caught live
            // by llmprobe cache-hit-reported on Qwen3-Coder, 2026-06-10).
            if (config.full_attention_interval > 0) {
                ssm_entries = ml.ssm_entries;
            } else {
                for (ml.ssm_entries) |*e| {
                    _ = mlx.mlx_array_free(e.conv_state);
                    _ = mlx.mlx_array_free(e.ssm_state);
                }
                allocator.free(ml.ssm_entries);
            }
        } else {
            const sl = try initStandardLayers(allocator, config, weights, &name_buf, s);
            layers = sl.layers;
            moe_owned_bf16 = sl.owned_bf16; // reuse the same deinit-tracked owned list
        }

        const bits_cache: BitsCache = .{};

        // LFM2: load embedding norm
        var embedding_norm_w: ?mlx.mlx_array = null;
        if (config.has_embedding_norm) {
            embedding_norm_w = getWeightFmtOpt(weights, &name_buf, "{s}.embedding_norm.weight", prefix);
        }

        // Gemma 4: load PLE global weights
        var ple_emb_w = mlx.mlx_array_new();
        var ple_emb_s = mlx.mlx_array_new();
        var ple_emb_b = mlx.mlx_array_new();
        var ple_proj_w_g = mlx.mlx_array_new();
        var ple_proj_s_g = mlx.mlx_array_new();
        var ple_proj_b_g = mlx.mlx_array_new();
        var ple_proj_norm = mlx.mlx_array_new();
        var ple_proj_quantized = false;
        if (config.hidden_size_per_layer_input > 0) {
            ple_emb_w = getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.weight", prefix);
            // Dense bf16: no scales/biases → null-ctx; dequantTake takes its dense path.
            ple_emb_s = if (config.quant_bits == 0) mlx.mlx_array_new() else getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.scales", prefix);
            ple_emb_b = if (config.quant_bits == 0)
                mlx.mlx_array_new()
            else if (bias_mandatory)
                getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.biases", prefix)
            else
                getWeightFmtOpt(weights, &name_buf, "{s}.embed_tokens_per_layer.biases", prefix) orelse mlx.mlx_array_new();
            ple_proj_w_g = getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.weight", prefix);
            // per_layer_model_projection may be unquantized (no scales/biases)
            if (getWeightFmtOpt(weights, &name_buf, "{s}.per_layer_model_projection.scales", prefix)) |sc| {
                ple_proj_s_g = sc;
                ple_proj_b_g = if (bias_mandatory)
                    getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.biases", prefix)
                else
                    getWeightFmtOpt(weights, &name_buf, "{s}.per_layer_model_projection.biases", prefix) orelse mlx.mlx_array_new();
                ple_proj_quantized = true;
            }
            ple_proj_norm = getWeightFmt(weights, &name_buf, "{s}.per_layer_projection_norm.weight", prefix);
        }

        // Gemma 4: logit softcapping scalar
        var softcap_scalar: ?mlx.mlx_array = null;
        if (config.final_logit_softcapping > 0) {
            softcap_scalar = bf16Scalar(config.final_logit_softcapping, s);
        }

        // Gemma 4: v_norm weights (parameter-free: ones vectors)
        var v_norm_weight: ?mlx.mlx_array = null;
        var v_norm_weight_global: ?mlx.mlx_array = null;
        if (config.has_v_norm) {
            const one_val = bf16Scalar(1.0, s);
            defer _ = mlx.mlx_array_free(one_val);
            const hd_shape = [_]c_int{@intCast(config.head_dim)};
            v_norm_weight = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_full(&v_norm_weight.?, &hd_shape, 1, one_val, .bfloat16, s));
            if (config.global_head_dim > 0 and config.global_head_dim != config.head_dim) {
                const ghd_shape = [_]c_int{@intCast(config.global_head_dim)};
                v_norm_weight_global = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_full(&v_norm_weight_global.?, &ghd_shape, 1, one_val, .bfloat16, s));
            }
        }

        // DiffusionGemma: self-conditioning module (the only diffusion-specific
        // decoder weights) + a hidden-sized ones vector for its scale-free
        // post_norm (the canvas embeddings pass through it even with a zero
        // conditioning signal — i.e. they are always RMS-normalized pre-layer-0).
        var self_cond: ?SelfCondWeights = null;
        var ones_hidden: ?mlx.mlx_array = null;
        if (config.isDiffusion()) {
            self_cond = .{
                .pre_norm = getWeightFmt(weights, &name_buf, "{s}.self_conditioning.pre_norm.weight", prefix),
                .gate_w = getWeightFmt(weights, &name_buf, "{s}.self_conditioning.gate_proj.weight", prefix),
                .gate_s = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.gate_proj.scales", prefix) orelse mlx.mlx_array_new(),
                .gate_b = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.gate_proj.biases", prefix) orelse mlx.mlx_array_new(),
                .up_w = getWeightFmt(weights, &name_buf, "{s}.self_conditioning.up_proj.weight", prefix),
                .up_s = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.up_proj.scales", prefix) orelse mlx.mlx_array_new(),
                .up_b = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.up_proj.biases", prefix) orelse mlx.mlx_array_new(),
                .down_w = getWeightFmt(weights, &name_buf, "{s}.self_conditioning.down_proj.weight", prefix),
                .down_s = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.down_proj.scales", prefix) orelse mlx.mlx_array_new(),
                .down_b = getWeightFmtOpt(weights, &name_buf, "{s}.self_conditioning.down_proj.biases", prefix) orelse mlx.mlx_array_new(),
            };
            const one_val = bf16Scalar(1.0, s);
            defer _ = mlx.mlx_array_free(one_val);
            const h_shape = [_]c_int{@intCast(config.hidden_size)};
            ones_hidden = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_full(&ones_hidden.?, &h_shape, 1, one_val, .bfloat16, s));
        }

        // Gemma 4: proportional RoPE frequencies for global/full attention layers
        // freqs = factor * base^(arange(0, rotated_dims, 2) / full_dims)
        // padded with inf for non-rotated dimensions
        var rope_freqs_global: ?mlx.mlx_array = null;
        if (config.rope_proportional) {
            const ghd: u32 = if (config.global_head_dim > 0) config.global_head_dim else config.head_dim;
            const rotated_dims: u32 = @intFromFloat(@as(f32, @floatFromInt(ghd)) * config.partial_rotary_factor_global);
            const n_rotated: u32 = rotated_dims / 2;
            const n_pad: u32 = (ghd - rotated_dims) / 2;
            const total: u32 = n_rotated + n_pad;

            const freq_shape = [_]c_int{@intCast(total)};
            var freqs_arr = mlx.mlx_array_new();

            // Compute rotated part: factor * base^(arange(0, rotated_dims, 2) / ghd)
            var arange_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(arange_arr);
            try mlx.check(mlx.mlx_arange(&arange_arr, 0, @floatFromInt(rotated_dims), 2, .float32, s));

            const ghd_scalar = mlx.mlx_array_new_float(@floatFromInt(ghd));
            defer _ = mlx.mlx_array_free(ghd_scalar);
            var exponents = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(exponents);
            try mlx.check(mlx.mlx_divide(&exponents, arange_arr, ghd_scalar, s));

            const base_scalar = mlx.mlx_array_new_float(config.rope_theta);
            defer _ = mlx.mlx_array_free(base_scalar);
            var base_pow = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(base_pow);
            try mlx.check(mlx.mlx_power(&base_pow, base_scalar, exponents, s));

            if (config.rope_proportional_factor != 1.0) {
                const factor_scalar = mlx.mlx_array_new_float(config.rope_proportional_factor);
                defer _ = mlx.mlx_array_free(factor_scalar);
                var scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&scaled, base_pow, factor_scalar, s));
                _ = mlx.mlx_array_free(base_pow);
                base_pow = scaled;
            }

            if (n_pad > 0) {
                // Pad with inf for non-rotated dims
                const pad_shape = [_]c_int{@intCast(n_pad)};
                const inf_val = mlx.mlx_array_new_float(std.math.inf(f32));
                defer _ = mlx.mlx_array_free(inf_val);
                var inf_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(inf_arr);
                try mlx.check(mlx.mlx_full(&inf_arr, &pad_shape, 1, inf_val, .float32, s));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                _ = mlx.mlx_vector_array_append_value(vec, base_pow);
                _ = mlx.mlx_vector_array_append_value(vec, inf_arr);
                try mlx.check(mlx.mlx_concatenate_axis(&freqs_arr, vec, 0, s));
            } else {
                try mlx.check(mlx.mlx_reshape(&freqs_arr, base_pow, &freq_shape, 1, s));
            }
            rope_freqs_global = freqs_arr;
        }

        // Laguna YaRN (full-attention layers): precompute the mlx_fast_rope
        // denominator array + the mscale vector (attention_factor on rotated
        // dims, 1.0 on the pass-through tail). Sliding layers use default RoPE.
        var rope_freqs_yarn: ?mlx.mlx_array = null;
        var yarn_mscale: ?mlx.mlx_array = null;
        if (config.rope_yarn) {
            const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(config.head_dim)) * config.partial_rotary_factor_global);
            const half: usize = rotary_dim / 2;
            const freqs_f64 = try allocator.alloc(f64, half);
            defer allocator.free(freqs_f64);
            computeYarnFreqs(
                freqs_f64,
                config.head_dim,
                config.partial_rotary_factor_global,
                config.rope_theta,
                config.yarn_factor,
                config.yarn_beta_fast,
                config.yarn_beta_slow,
                config.yarn_orig_max_pos,
            );
            const freqs_f32 = try allocator.alloc(f32, half);
            defer allocator.free(freqs_f32);
            for (freqs_f64, 0..) |v, i| freqs_f32[i] = @floatCast(v);
            const fshape = [_]c_int{@intCast(half)};
            rope_freqs_yarn = mlx.mlx_array_new_data(freqs_f32.ptr, &fshape, 1, .float32);
            // mscale vector [head_dim]: af on rotary dims, 1.0 on the tail.
            // Multiplying the post-rope q/k by this is exactly the reference's
            // cos/sin *= attention_factor (rotated dims only, pass dims untouched).
            const mscale_f32 = try allocator.alloc(f32, config.head_dim);
            defer allocator.free(mscale_f32);
            for (mscale_f32, 0..) |*m, i| m.* = if (i < rotary_dim) config.yarn_attention_factor else 1.0;
            const mshape = [_]c_int{@intCast(config.head_dim)};
            yarn_mscale = mlx.mlx_array_new_data(mscale_f32.ptr, &mshape, 1, .float32);
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, rope_freqs_yarn.?);
            _ = mlx.mlx_vector_array_append_value(eval_vec, yarn_mscale.?);
            try mlx.check(mlx.mlx_eval(eval_vec));
        }

        // Batch eval all weights
        {
            const eval_start = std.Io.Timestamp.now(io, .awake);
            const all_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(all_vec);

            _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
            // Dense bf16 models have null-ctx embedding/lm_head scales/biases —
            // appending a null array aborts in mlx (vector.cpp "non-empty" guard).
            if (emb_s_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_s_arr);
            if (emb_b_arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, emb_b_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_w);
            if (lm_head_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_s);
            if (lm_head_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_b);
            if (dense0_w.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(all_vec, dense0_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, dense1_w);
                if (dense0_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense0_s);
                if (dense0_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense0_b);
                if (dense1_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense1_s);
                if (dense1_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, dense1_b);
            }

            if (moe_layers) |ml| {
                for (ml) |lw| {
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                    if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.pre_ff_norm_2) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.post_ff_norm_1) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.post_ff_norm_2) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.layer_scalar) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    appendHybridMlpWeights(all_vec, &lw.mlp);
                    if (lw.shared_mlp) |smlp| {
                        inline for (comptime structFields(DenseMlpWeights)) |field| {
                            const arr = @field(smlp, field.name);
                            if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, arr);
                        }
                    }
                    switch (lw.attn) {
                        .full => |fa| appendFullAttnWeights(all_vec, &fa),
                        .linear => |la| appendLinearAttnWeights(all_vec, &la),
                    }
                }
            } else {
                for (layers) |lw| {
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                    if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.q_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.k_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    // Dense bf16 layers carry null-ctx scales/biases — skip them so
                    // the eval batch doesn't get a null array (mlx aborts on append).
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_w);
                    if (lw.q_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_s);
                    if (lw.q_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_b);
                    if (lw.k_w.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_w);
                    if (lw.k_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_s);
                    if (lw.k_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_b);
                    if (lw.v_w.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_w);
                    if (lw.v_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_s);
                    if (lw.v_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_b);
                    // Additive qkv biases (Qwen2) — empty-ctx for archs without them.
                    if (lw.q_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_bias);
                    if (lw.k_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_bias);
                    if (lw.v_bias.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_bias);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_w);
                    if (lw.o_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_s);
                    if (lw.o_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_b);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_w);
                    if (lw.gate_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_s);
                    if (lw.gate_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_b);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_w);
                    if (lw.up_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_s);
                    if (lw.up_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_b);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_w);
                    if (lw.down_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_s);
                    if (lw.down_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_b);
                    if (lw.layer_scalar) |ls| _ = mlx.mlx_vector_array_append_value(all_vec, ls);
                    if (lw.ple_gate_w) |w| _ = mlx.mlx_vector_array_append_value(all_vec, w);
                    if (lw.ple_gate_s) |sc| if (sc.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                    };
                    if (lw.ple_gate_b) |bi| if (bi.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(all_vec, bi);
                    };
                    if (lw.ple_proj_w) |w| _ = mlx.mlx_vector_array_append_value(all_vec, w);
                    if (lw.ple_proj_s) |sc| if (sc.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                    };
                    if (lw.ple_proj_b) |bi| if (bi.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(all_vec, bi);
                    };
                    if (lw.ple_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                }
            }

            try mlx.check(mlx.mlx_eval(all_vec));
            const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
            log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
        }

        const profile_head_shape = mlx.getShape(lm_head_w);
        const profile_head_n: c_int = if (profile_head_shape.len == 2) profile_head_shape[0] else 0;
        const mtp_uniform_affine_trunk = mtpNaxCalibratedModelFrom(&config, profile_head_n) and
            config.quant_bits == 4 and
            config.quant_group_size == 64 and
            config.quant_mode == .affine and
            mtpNaxAffineProjectionMatches(&config, emb_w, emb_s_arr, emb_b_arr, config.hidden_size, config.vocab_size) and
            mtpNaxUniformAffineTrunkFrom(&config, moe_layers);
        const mtp_oqe_affine_trunk = mtpNaxCalibratedModelFrom(&config, profile_head_n) and
            config.quant_bits == 4 and
            config.quant_group_size == 64 and
            config.quant_mode == .affine and
            mtpNaxAffineProjectionMatches(&config, emb_w, emb_s_arr, emb_b_arr, config.hidden_size, config.vocab_size) and
            mtpNaxOqeAffineTrunkFrom(&config, moe_layers);

        return .{
            .config = config,
            .cache = cache,
            .s = s,
            .allocator = allocator,
            .emb_w = emb_w,
            .emb_s = emb_s_arr,
            .emb_b = emb_b_arr,
            .emb_scale = emb_scale,
            .final_norm = final_norm,
            .lm_head_w = lm_head_w,
            .lm_head_s = lm_head_s,
            .lm_head_b = lm_head_b,
            .dense0_w = dense0_w,
            .dense0_s = dense0_s,
            .dense0_b = dense0_b,
            .dense1_w = dense1_w,
            .dense1_s = dense1_s,
            .dense1_b = dense1_b,
            .layers = layers,
            .owns_lm_head = owns_lm_head,
            .owns_norms = config.norm_has_offset,
            .gelu_coeff = if (need_gelu) bf16Scalar(0.7978845608028654, s) else null,
            .gelu_inner = if (need_gelu) bf16Scalar(0.044715, s) else null,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .three = if (need_gelu) bf16Scalar(3.0, s) else null,
            .neg_one = if (need_silu) bf16Scalar(-1.0, s) else null,
            .ple_emb_w = ple_emb_w,
            .ple_emb_s = ple_emb_s,
            .ple_emb_b = ple_emb_b,
            .ple_proj_w = ple_proj_w_g,
            .ple_proj_s = ple_proj_s_g,
            .ple_proj_b = ple_proj_b_g,
            .ple_proj_norm = ple_proj_norm,
            .ple_proj_quantized = ple_proj_quantized,
            .softcap_scalar = softcap_scalar,
            .v_norm_weight = v_norm_weight,
            .v_norm_weight_global = v_norm_weight_global,
            .self_cond = self_cond,
            .ones_hidden = ones_hidden,
            .rope_freqs_global = rope_freqs_global,
            .rope_freqs_yarn = rope_freqs_yarn,
            .yarn_mscale = yarn_mscale,
            .bert_layers = null,
            .bert_pos_w = mlx.mlx_array_new(),
            .bert_pos_s = mlx.mlx_array_new(),
            .bert_pos_b = mlx.mlx_array_new(),
            .bert_toktype_w = mlx.mlx_array_new(),
            .bert_toktype_s = mlx.mlx_array_new(),
            .bert_toktype_b = mlx.mlx_array_new(),
            .bert_emb_norm_w = mlx.mlx_array_new(),
            .bert_emb_norm_b = mlx.mlx_array_new(),
            .moe_layers = moe_layers,
            .ssm_entries = ssm_entries,
            .moe_seq_offset = 0,
            .moe_owned_bf16 = moe_owned_bf16,
            .hybrid_layers = hybrid_layers,
            .embedding_norm = embedding_norm_w,
            .prompt_cache = null,
            .bits_cache = bits_cache,
            .mtp_uniform_affine_trunk = mtp_uniform_affine_trunk,
            .mtp_oqe_affine_trunk = mtp_oqe_affine_trunk,
        };
    }

    /// Reset all caches for a new request (KV cache + SSM state for MoE).
    pub fn resetCache(self: *Transformer) !void {
        const prev_config = self.cache.config;
        self.cache.deinit();
        self.cache = try KVCache.initWithConfigAndHeadDim(self.allocator, self.config.num_hidden_layers, prev_config, self.config.head_dim);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                e.conv_state = mlx.mlx_array_new();
                e.ssm_state = mlx.mlx_array_new();
                e.initialized = false;
            }
        }
        self.moe_seq_offset = 0;
    }

    /// Try to restore state from prompt cache if the cached tokens are an exact
    /// prefix of new_ids. Returns the number of matched (restored) tokens, or 0
    /// if the cache missed and a full reset was performed.
    pub fn tryRestoreCache(self: *Transformer, new_ids: []const u32) !usize {
        const pc = self.prompt_cache orelse {
            try self.resetCache();
            return 0;
        };

        const match_limit = @min(pc.tokens.len, new_ids.len);
        var matched: usize = 0;
        while (matched < match_limit) : (matched += 1) {
            if (pc.tokens[matched] != new_ids[matched]) break;
        }

        if (matched < pc.tokens.len or matched >= new_ids.len) {
            try self.resetCache();
            return 0;
        }

        // Full prefix match with tokens remaining — restore cached state.
        const prev_config = self.cache.config;
        self.cache.deinit();
        self.cache = try KVCache.initWithConfigAndHeadDim(self.allocator, self.config.num_hidden_layers, prev_config, self.config.head_dim);
        self.cache.step = pc.kv_step;
        for (pc.kv_entries, 0..) |src, i| {
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values, src.values));
                if (prev_config.scheme == .affine) {
                    try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values_biases, src.values_biases));
                }
                self.cache.entries[i].initialized = true;
                self.cache.entries[i].offset = pc.offsets[i];
                // *_view fields left as mlx_array_new() — recreated on next update()
            }
        }

        if (pc.ssm_entries) |ssm_src| {
            if (self.ssm_entries) |ssm_dst| {
                for (ssm_src, ssm_dst) |src, *dst| {
                    _ = mlx.mlx_array_free(dst.conv_state);
                    _ = mlx.mlx_array_free(dst.ssm_state);
                    dst.conv_state = mlx.mlx_array_new();
                    dst.ssm_state = mlx.mlx_array_new();
                    dst.initialized = src.initialized;
                    // Per-field null guard — LFM2 gated_conv layers fill only
                    // conv_state, never ssm_state, even after initialization.
                    if (src.conv_state.ctx != null) {
                        try mlx.check(mlx.mlx_array_set(&dst.conv_state, src.conv_state));
                    }
                    if (src.ssm_state.ctx != null) {
                        try mlx.check(mlx.mlx_array_set(&dst.ssm_state, src.ssm_state));
                    }
                }
            }
        }

        self.moe_seq_offset = pc.moe_seq_offset;
        return matched;
    }

    /// Snapshot the current KV cache + SSM state so the next request can reuse
    /// them if its prompt starts with the same token prefix.
    pub fn savePromptCache(self: *Transformer, prompt_ids: []const u32) void {
        if (self.prompt_cache) |*pc| pc.deinit();
        self.prompt_cache = null;

        const tokens = self.allocator.dupe(u32, prompt_ids) catch return;
        const num_layers = self.cache.entries.len;
        const kv = self.allocator.alloc(KVCacheEntry, num_layers) catch {
            self.allocator.free(tokens);
            return;
        };
        const offsets = self.allocator.alloc(usize, num_layers) catch {
            self.allocator.free(tokens);
            self.allocator.free(kv);
            return;
        };
        const scheme = self.cache.config.scheme;
        for (self.cache.entries, kv, 0..) |src, *dst, i| {
            dst.* = newEmptyKVEntry();
            dst.offset = src.offset;
            if (src.initialized) {
                _ = mlx.mlx_array_set(&dst.keys, src.keys);
                _ = mlx.mlx_array_set(&dst.values, src.values);
                if (scheme == .affine) {
                    _ = mlx.mlx_array_set(&dst.keys_scales, src.keys_scales);
                    _ = mlx.mlx_array_set(&dst.keys_biases, src.keys_biases);
                    _ = mlx.mlx_array_set(&dst.values_scales, src.values_scales);
                    _ = mlx.mlx_array_set(&dst.values_biases, src.values_biases);
                }
            }
            dst.initialized = src.initialized;
            offsets[i] = src.offset;
        }

        var ssm: ?[]SSMCacheEntry = null;
        if (self.ssm_entries) |entries| {
            const ssm_copy = self.allocator.alloc(SSMCacheEntry, entries.len) catch return;
            for (entries, ssm_copy) |src, *dst| {
                dst.conv_state = mlx.mlx_array_new();
                dst.ssm_state = mlx.mlx_array_new();
                dst.initialized = src.initialized;
                // Per-field null guard — LFM2 gated_conv layers fill only
                // conv_state, never ssm_state, even when `initialized==true`.
                if (src.conv_state.ctx != null) {
                    _ = mlx.mlx_array_set(&dst.conv_state, src.conv_state);
                }
                if (src.ssm_state.ctx != null) {
                    _ = mlx.mlx_array_set(&dst.ssm_state, src.ssm_state);
                }
            }
            ssm = ssm_copy;
        }

        self.prompt_cache = .{
            .tokens = tokens,
            .kv_entries = kv,
            .offsets = offsets,
            .kv_step = self.cache.step,
            .ssm_entries = ssm,
            .moe_seq_offset = self.moe_seq_offset,
            .allocator = self.allocator,
        };
    }

    /// Create a compiled version of the forward pass for faster decode.
    pub fn compileForward(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &forwardClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, false);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_forward = compiled;
            log.info("Forward pass compiled (Metal kernel fusion enabled)\n", .{});
        } else {
            log.warn("Forward compilation failed, using uncompiled path\n", .{});
        }
    }

    /// Compile GELU activation for kernel fusion.
    /// Fuses 8 separate ops into 1 GPU kernel, matching mlx-lm's @mx.compile behavior.
    pub fn compileGelu(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &geluClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true); // shapeless=true
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_gelu = compiled;
            log.info("GELU compiled (kernel fusion enabled)\n", .{});
        } else {
            log.warn("GELU compilation failed, using uncompiled path\n", .{});
        }
    }

    /// Compile GeGLU: gelu(gate) * up → single fused kernel.
    pub fn compileGeglu(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &gegluClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_geglu = compiled;
            log.info("GeGLU compiled (kernel fusion enabled)\n", .{});
        }
    }

    fn gegluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var gate = mlx.mlx_array_new();
        var up = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&gate, input, 0) != 0) return -1;
        if (mlx.mlx_vector_array_get(&up, input, 1) != 0) {
            _ = mlx.mlx_array_free(gate);
            return -1;
        }
        defer _ = mlx.mlx_array_free(gate);
        defer _ = mlx.mlx_array_free(up);

        // geglu(gate, up) = gelu_approx(gate) * up
        const activated = self.geluUncompiled(gate) catch return -1;
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        mlx.check(mlx.mlx_multiply(&result, activated, up, self.s)) catch return -1;

        const out_arr = [_]mlx.mlx_array{result};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(result);
        return 0;
    }

    /// Compile logit softcap: tanh(x/cap) * cap → single fused kernel.
    pub fn compileSoftcap(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &softcapClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_softcap = compiled;
            log.info("Softcap compiled (kernel fusion enabled)\n", .{});
        }
    }

    fn softcapClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var x = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(x);

        const cap = self.softcap_scalar orelse return -1;
        // tanh(x / cap) * cap
        var scaled = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&scaled, x, cap, self.s)) catch return -1;
        defer _ = mlx.mlx_array_free(scaled);
        var tanh_val = mlx.mlx_array_new();
        mlx.check(mlx.mlx_tanh(&tanh_val, scaled, self.s)) catch return -1;
        defer _ = mlx.mlx_array_free(tanh_val);
        var result = mlx.mlx_array_new();
        mlx.check(mlx.mlx_multiply(&result, tanh_val, cap, self.s)) catch return -1;

        const out_arr = [_]mlx.mlx_array{result};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(result);
        return 0;
    }

    fn geluClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var x = mlx.mlx_array_new();
        const get_rc = mlx.mlx_vector_array_get(&x, input, 0);
        if (get_rc != 0) return get_rc;
        defer _ = mlx.mlx_array_free(x);

        // gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
        const result = self.geluUncompiled(x) catch return -1;

        const out_arr = [_]mlx.mlx_array{result};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(result);
        return 0;
    }

    /// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
    /// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel.
    /// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
    pub fn compileGdnGate(self: *Transformer) void {
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &gdnGateClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, true);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_gdn_gate = compiled;
            log.info("GDN gate compiled (kernel fusion enabled)\n", .{});
        }
    }

    fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var A_log = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(A_log);
        var a = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
        defer _ = mlx.mlx_array_free(a);
        var dt_bias = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
        defer _ = mlx.mlx_array_free(dt_bias);

        const g = gdnGateChain(A_log, a, dt_bias, self.s) catch return -1;
        const out_arr = [_]mlx.mlx_array{g};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(g);
        return 0;
    }

    /// Apply the compiled GDN gate closure if available, else the raw chain.
    /// Returns owned g (bf16, shape of `a`).
    fn computeGdnGate(self: *const Transformer, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_gdn_gate) |compiled| {
            const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            if (mlx.mlx_vector_array_size(out_vec) == 1) {
                var g = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
                return g;
            }
        }
        return gdnGateChain(A_log, a, dt_bias, self.s);
    }

    /// Compile MoE routing (negate→argpartition→slice→softmax→take_along_axis→sum→expand→divide)
    /// into a single fused kernel. Input: router_logits. Outputs: inds, norm_scores.
    /// shapeless=false: slice bounds derive from input ndim, so the closure must
    /// re-trace per input shape. MoE inference only sees two shapes in practice
    /// (decode seq_len=1, prefill seq_len=N), so the trace cost amortizes after
    /// the first prefill + first decode.
    pub fn compileMoeRouting(self: *Transformer) void {
        if (self.config.moe_sigmoid_router) {
            // hy_v3: the sigmoid+bias chain takes a second input (the per-layer
            // expert_bias weight), so it compiles as its own closure.
            const raw_closure = mlx.mlx_closure_new_func_payload(
                &hy3RoutingClosureCallback,
                @ptrCast(self),
                null,
            );
            var compiled = mlx.mlx_closure{ .ctx = null };
            const rc = mlx.mlx_compile(&compiled, raw_closure, false);
            _ = mlx.mlx_closure_free(raw_closure);
            if (rc == 0 and compiled.ctx != null) {
                self.compiled_hy3_routing = compiled;
                log.info("MoE routing compiled (hy3 sigmoid variant)\n", .{});
            }
            return;
        }
        const raw_closure = mlx.mlx_closure_new_func_payload(
            &moeRoutingClosureCallback,
            @ptrCast(self),
            null,
        );
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw_closure, false);
        _ = mlx.mlx_closure_free(raw_closure);
        if (rc == 0 and compiled.ctx != null) {
            self.compiled_moe_routing = compiled;
            log.info("MoE routing compiled (kernel fusion enabled)\n", .{});
        }
    }

    /// Result type for the MoE routing helpers. Both fields are owned arrays —
    /// caller is responsible for freeing them.
    const MoeRouting = struct { inds: mlx.mlx_array, norm_scores: mlx.mlx_array };

    /// Pure subgraph for MoE routing. Inputs:
    ///   [0] router_logits — shape [..., num_experts]
    /// Outputs:
    ///   [0] inds         — shape [..., K], int32 expert indices (top-K)
    ///   [1] norm_scores  — shape [..., K], renormalized top-K softmax weights
    ///
    /// The sigma-MoE per-expert-scale path stays outside the closure because it
    /// branches on per-layer weights at model-load time.
    fn moeRoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        const k: c_int = @intCast(self.config.num_experts_per_tok);

        var router_logits = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(router_logits);

        const inds_norm = self.moeRoutingUncompiled(router_logits, k) catch return -1;
        defer _ = mlx.mlx_array_free(inds_norm.inds);
        defer _ = mlx.mlx_array_free(inds_norm.norm_scores);

        const out_arr = [_]mlx.mlx_array{ inds_norm.inds, inds_norm.norm_scores };
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
        return 0;
    }

    /// Reference implementation of the MoE routing chain (used both as fallback
    /// and as the body the compiled closure traces). Returns owned `inds` +
    /// `norm_scores` arrays — caller must free both.
    fn moeRoutingUncompiled(self: *const Transformer, router_logits: mlx.mlx_array, k: c_int) !MoeRouting {
        return moeRoutingChain(router_logits, k, self.s);
    }

    /// Hy3 sigmoid routing closure body. Inputs:
    ///   [0] router_logits — [..., num_experts]
    ///   [1] expert_bias   — f32 [num_experts]
    /// route_norm / route_scale / k are model-constant (from config), so they
    /// bake into the trace.
    fn hy3RoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        const k: c_int = @intCast(self.config.num_experts_per_tok);

        var router_logits = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
        defer _ = mlx.mlx_array_free(router_logits);
        var expert_bias = mlx.mlx_array_new();
        if (mlx.mlx_vector_array_get(&expert_bias, input, 1) != 0) return -1;
        defer _ = mlx.mlx_array_free(expert_bias);

        const routed = hy3RoutingChain(
            router_logits,
            expert_bias,
            k,
            self.config.moe_route_norm,
            self.config.router_scaling_factor,
            self.s,
        ) catch return -1;
        defer _ = mlx.mlx_array_free(routed.inds);
        defer _ = mlx.mlx_array_free(routed.norm_scores);

        const out_arr = [_]mlx.mlx_array{ routed.inds, routed.norm_scores };
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
        return 0;
    }

    /// Hy3 sigmoid+bias routing — compiled closure when available, else the
    /// direct chain. Returns owned `inds` + `norm_scores`.
    fn computeHy3Routing(self: *const Transformer, router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array) !MoeRouting {
        if (self.compiled_hy3_routing) |compiled| {
            const in_arr = [_]mlx.mlx_array{ router_logits, expert_bias };
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 2);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);

            var inds = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(inds);
            try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
            var norm_scores = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(norm_scores);
            try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
            return .{ .inds = inds, .norm_scores = norm_scores };
        }
        return hy3RoutingChain(
            router_logits,
            expert_bias,
            @intCast(self.config.num_experts_per_tok),
            self.config.moe_route_norm,
            self.config.router_scaling_factor,
            self.s,
        );
    }

    /// Apply the compiled MoE routing closure if available, else fall back.
    /// Returns owned `inds` + `norm_scores` — caller must free both.
    fn computeMoeRouting(self: *const Transformer, router_logits: mlx.mlx_array) !MoeRouting {
        const k: c_int = @intCast(self.config.num_experts_per_tok);
        if (self.compiled_moe_routing) |compiled| {
            const in_arr = [_]mlx.mlx_array{router_logits};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);

            var inds = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(inds);
            try mlx.check(mlx.mlx_vector_array_get(&inds, out_vec, 0));
            var norm_scores = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(norm_scores);
            try mlx.check(mlx.mlx_vector_array_get(&norm_scores, out_vec, 1));
            return .{ .inds = inds, .norm_scores = norm_scores };
        }
        return self.moeRoutingUncompiled(router_logits, k);
    }

    fn forwardClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *Transformer = @ptrCast(@alignCast(payload.?));
        var token_ids = mlx.mlx_array_new();
        const get_rc = mlx.mlx_vector_array_get(&token_ids, input, 0);
        if (get_rc != 0) return get_rc;
        defer _ = mlx.mlx_array_free(token_ids);

        const logits = self.forward(token_ids) catch return -1;

        const out_arr = [_]mlx.mlx_array{logits};
        res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
        _ = mlx.mlx_array_free(logits);
        return 0;
    }

    /// Forward pass using compiled closure if available, falling back to regular.
    pub fn forwardCompiled(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_forward) |compiled| {
            const in_arr = [_]mlx.mlx_array{token_ids};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);

            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);

            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        return self.forward(token_ids);
    }

    pub fn deinit(self: *Transformer) void {
        if (self.compiled_forward) |cf| _ = mlx.mlx_closure_free(cf);
        if (self.compiled_gelu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_geglu) |cg| _ = mlx.mlx_closure_free(cg);
        if (self.compiled_softcap) |cs| _ = mlx.mlx_closure_free(cs);
        if (self.compiled_moe_routing) |cmr| _ = mlx.mlx_closure_free(cmr);
        if (self.compiled_hy3_routing) |chr| _ = mlx.mlx_closure_free(chr);
        if (self.compiled_gdn_gate) |cgg| _ = mlx.mlx_closure_free(cgg);
        if (self.gdn_ones_w) |w| _ = mlx.mlx_array_free(w);
        if (self.gdn_q_scale) |q| _ = mlx.mlx_array_free(q);
        if (self.gdn_k_scale) |k| _ = mlx.mlx_array_free(k);
        if (self.ones_hidden) |o| _ = mlx.mlx_array_free(o);
        if (self.rope_freqs_yarn) |f| _ = mlx.mlx_array_free(f);
        if (self.yarn_mscale) |m| _ = mlx.mlx_array_free(m);
        if (self.prompt_cache) |*pc| pc.deinit();
        self.cache.deinit();
        if (self.emb_scale) |es| _ = mlx.mlx_array_free(es);
        if (self.owns_norms) _ = mlx.mlx_array_free(self.final_norm);
        if (self.gelu_coeff) |g| _ = mlx.mlx_array_free(g);
        if (self.gelu_inner) |g| _ = mlx.mlx_array_free(g);
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        if (self.three) |t| _ = mlx.mlx_array_free(t);
        if (self.neg_one) |n| _ = mlx.mlx_array_free(n);
        for (self.layers) |lw| {
            if (self.owns_norms) {
                _ = mlx.mlx_array_free(lw.input_norm);
                _ = mlx.mlx_array_free(lw.post_attn_norm);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.q_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.k_norm) |n| _ = mlx.mlx_array_free(n);
            }
        }
        self.allocator.free(self.layers);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                ssmFreeSpecCapture(e);
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            self.allocator.free(entries);
        }
        if (self.moe_layers) |ml| self.allocator.free(ml);
        if (self.moe_owned_bf16) |arrs| {
            for (arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(arrs);
        }
        // self.s is the thread's default GPU stream (not owned by us) — don't free it.
        _ = mlx.mlx_stream_free(self.s);
        // The free above is a no-op on the default stream's wrapper but we keep it for symmetry
        // with the Zig copy of the mlx_stream struct that init handed us.
    }

    /// Re-bind `self.s` to the *current* thread's default GPU stream. Must be called from any
    /// thread that's about to use this Transformer for inference, since mlx 0.31.2 made streams
    /// thread-local (a stream created on thread A is invisible to thread B).
    pub fn useCurrentThreadStream(self: *Transformer) void {
        self.s = mlx.gpuStream();
    }

    // ── Core ops ──

    inline fn qmatmul(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
        // Resolve (bits, group_size, mode) per weight. Most weights inherit the
        // global config; per-weight overrides (mixed-precision checkpoints, e.g.
        // affine 8-bit shared MLP inside an nvfp4 QAT model) are detected on
        // first touch — x's inner dim pins (bits, group_size) exactly.
        const qp = self.quantParamsHinted(w, sc, lastDim(x));
        return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
    }

    /// Final logits projection. For dense bf16 the lm_head weight is [vocab, hidden]
    /// and is NOT pre-transposed at load (when tied it aliases emb_w, which must stay
    /// [vocab, hidden] for the embedding lookup), so we project via a lazy transposed
    /// view. Quantized models fall through to the standard gather/qmm path unchanged.
    inline fn lmHeadProject(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (self.lm_head_s.ctx == null) {
            const wt = try transposeBf16Weight(self.lm_head_w, self.s); // [hidden, vocab] view
            defer _ = mlx.mlx_array_free(wt);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&result, x, wt, self.s));
            return result;
        }
        return self.qmatmul(x, self.lm_head_w, self.lm_head_s, self.lm_head_b);
    }

    /// Resolve per-weight quant params without an activation hint — for callers
    /// like the embedding gather whose tensors follow the model-wide scheme.
    inline fn quantParamsFor(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array) QuantParams {
        return self.quantParamsHinted(w, sc, null);
    }

    /// Resolve per-weight (bits, group_size, mode) with a lazy cache keyed by
    /// the scales array pointer. First touch computes params from the scales
    /// dtype + shapes (~4 FFI calls); subsequent calls are a single pointer
    /// compare. `in_dim` is the weight's input dimension when the caller knows
    /// it (activation inner dim) — required to disambiguate (bits, group_size)
    /// for affine-override tensors inside a non-affine model.
    /// Thread-safety: generation is single-threaded.
    inline fn quantParamsHinted(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
        const key_raw = sc.ctx orelse return .{
            .bits = self.config.quant_bits,
            .group_size = self.config.quant_group_size,
            .mode = self.config.quant_mode,
        };
        const cache = @constCast(&self.bits_cache);
        const start = BitsCache.slot(key_raw);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            const entry = cache.keys[idx];
            if (entry == key_raw) {
                return .{
                    .bits = cache.vals_bits[idx],
                    .group_size = @as(u32, cache.vals_gs_div8[idx]) * 8,
                    .mode = @enumFromInt(cache.vals_mode[idx]),
                };
            }
            if (entry == null) {
                const detected = computeQuantParams(&self.config, w, sc, in_dim);
                _ = cache.put(key_raw, detected);
                return detected;
            }
        }
        // Probe window saturated — fall through to direct detect, no cache write.
        return computeQuantParams(&self.config, w, sc, in_dim);
    }

    fn mtpNaxProfileEnabledForTrunk(self: *const Transformer, profiled_trunk: bool, mixed_bits: bool) bool {
        if (self.config.isMoe()) return false;
        if (self.config.hidden_size > std.math.maxInt(c_int)) return false;
        if (self.config.quant_bits != 4 or self.config.quant_mode != .affine) return false;
        if (self.config.quant_group_size != 64) return false;
        if (mixed_bits and !naxMixedBitsEnvEnabled()) return false;
        if (self.lm_head_w.ctx == null or
            self.lm_head_s.ctx == null or
            self.lm_head_b.ctx == null) return false;
        if (mlx.mlx_array_dtype(self.lm_head_w) != .uint32) return false;
        const K: c_int = @intCast(self.config.hidden_size);
        const w_shape = mlx.getShape(self.lm_head_w);
        if (w_shape.len != 2) return false;
        // The cost fit is intentionally narrower than kernel eligibility. It
        // was measured on the dense Qwen3.6-27B class; other NAX-capable
        // models retain auto cap 6 until their full-round surface is measured
        // (explicit --mtp-depth 7/8 still works there).
        const calibrated_model = mtpNaxCalibratedModelFrom(&self.config, w_shape[0]);
        const calibrated_head = mtpNaxAffineProjectionMatches(
            &self.config,
            self.lm_head_w,
            self.lm_head_s,
            self.lm_head_b,
            self.config.hidden_size,
            self.config.vocab_size,
        );

        return mtpNaxProfileEnabledFrom(.{
            .dense_model = true,
            .calibrated_model = calibrated_model,
            .profiled_affine_trunk = profiled_trunk,
            .model_quant = .{
                .bits = self.config.quant_bits,
                .group_size = self.config.quant_group_size,
                .mode = self.config.quant_mode,
            },
            .weight_present = self.lm_head_w.ctx != null,
            .packed_weight = calibrated_head,
            .scales_present = self.lm_head_s.ctx != null,
            .biases_present = self.lm_head_b.ctx != null,
            .quant = self.quantParamsHinted(self.lm_head_w, self.lm_head_s, self.config.hidden_size),
            .K = K,
            .N = w_shape[0],
            .packed_k = w_shape[1],
            .verify_on = verifyQmmEnabled(),
            .lane_on = naxLaneEnvEnabled(),
            .available = verifyQmmNaxAvailable(),
            .min_m = naxMinM(),
        });
    }

    /// Homogeneous q4/gs64 target profile used by the existing q4/q8-gs32
    /// MTP sidecar cost surfaces.
    pub fn mtpNaxProfileEnabled(self: *const Transformer) bool {
        return self.mtpNaxProfileEnabledForTrunk(self.mtp_uniform_affine_trunk, false);
    }

    /// Exact oQ4e mixed q4/q5/q6 target profile. The q5/q6 lane kill switch
    /// also revokes the cost profile so auto depth never assumes unavailable
    /// mixed-bit acceleration.
    pub fn mtpOqeNaxProfileEnabled(self: *const Transformer) bool {
        return self.mtpNaxProfileEnabledForTrunk(self.mtp_oqe_affine_trunk, true);
    }

    inline fn rmsNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
        return result;
    }

    inline fn layerNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&result, x, w, b, self.config.layer_norm_eps, self.s));
        return result;
    }

    inline fn qmatmulAddBias(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
        const mm = try self.qmatmul(x, w, sc, bi);
        defer _ = mlx.mlx_array_free(mm);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&result, mm, bias, self.s));
        return result;
    }

    /// `qmatmul`, plus an additive projection bias when `bias` is non-empty.
    /// Lets the standard (Llama-family) attention path support archs that carry
    /// additive qkv biases (Qwen2's `q/k/v_proj.bias`) without branching per
    /// arch — empty `bias` (qwen3/llama/mistral) is a plain `qmatmul`.
    inline fn qmatmulMaybeBias(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
        if (bias.ctx != null) return self.qmatmulAddBias(x, w, sc, bi, bias);
        return self.qmatmul(x, w, sc, bi);
    }

    fn embedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        const flat_shape = [_]c_int{batch * seq_len};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        var taken_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_w);
        try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));

        var emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb);
        if (self.emb_s.ctx == null) {
            // Dense bf16 embedding table: the gathered rows ARE the embeddings.
            try mlx.check(mlx.mlx_astype(&emb, taken_w, .bfloat16, self.s));
        } else {
            var taken_s = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_s);
            try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
            // Bias-less modes (nvfp4/mxfp4/mxfp8) have a null-ctx emb_b —
            // mlx_take_axis on a null handle aborts, so gather only when present.
            var taken_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_b);
            if (self.emb_b.ctx != null) {
                try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));
            }
            const emb_qp = self.quantParamsHinted(self.emb_w, self.emb_s, self.config.hidden_size);
            try mlx.check(mlx.mlx_dequantize(
                &emb,
                taken_w,
                taken_s,
                taken_b,
                mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
                mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
                emb_qp.mode.cstr(),
                .{}, // global_scale (null)
                .{ .value = .bfloat16, .has_value = true },
                self.s,
            ));
        }

        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var reshaped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(reshaped);
        try mlx.check(mlx.mlx_reshape(&reshaped, emb, &out_shape, 3, self.s));

        if (self.emb_scale) |scale| {
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, reshaped, scale, self.s));
            return scaled;
        }
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&result, reshaped));
        return result;
    }

    /// Dense [vocab, hidden] bf16 view of the embedding table. DiffusionGemma
    /// self-conditioning computes `probs @ table` — a plain matmul over a
    /// float table — so quantized checkpoints dequantize the WHOLE table once
    /// per generation request (mlx-vlm does the same; quantized_matmul with
    /// transpose=false is several times slower at this shape). ~1.5 GB
    /// transient for the 262K×2816 table. Caller frees. Dense bf16
    /// checkpoints return a refcount-share of the live table.
    pub fn dequantizedEmbedding(self: *Transformer) !mlx.mlx_array {
        if (self.emb_s.ctx == null) {
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&out, self.emb_w));
            return out;
        }
        const emb_qp = self.quantParamsHinted(self.emb_w, self.emb_s, self.config.hidden_size);
        var dense = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_dequantize(
            &dense,
            self.emb_w,
            self.emb_s,
            self.emb_b,
            mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
            emb_qp.mode.cstr(),
            .{}, // global_scale (null)
            .{ .value = .bfloat16, .has_value = true },
            self.s,
        ));
        return dense;
    }

    /// Splice vision embeddings into text embeddings at image_token_id positions.
    /// h: [B, seq_len, hidden] text embeddings
    /// token_ids: [B, seq_len] original token IDs
    /// vision_emb: [B, N_img, hidden] vision embeddings
    /// Returns new h with vision embeddings replacing image token positions.
    /// masked_scatter: replaces image token positions with vision features.
    /// Matches Python reference: cumsum-based indexing into flattened source.
    fn spliceVisionEmbeddings(self: *Transformer, h: mlx.mlx_array, token_ids: mlx.mlx_array, vision_emb: mlx.mlx_array, image_token_id: u32, audio_token_id: u32) !mlx.mlx_array {
        const h_shape = mlx.getShape(h);

        // mask = (token_ids == image_token_id) [| (token_ids == audio_token_id)].
        // Gemma 4 12B unified splices both modalities through this one channel:
        // the embedding tensor concatenates [vision rows ; audio rows] in the
        // same order the placeholder blocks were injected into the prompt, so a
        // single sequence-order scatter lands each row in its slot.
        const img_id_arr = mlx.mlx_array_new_int(@intCast(image_token_id));
        defer _ = mlx.mlx_array_free(img_id_arr);
        var mask_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_2d);
        try mlx.check(mlx.mlx_equal(&mask_2d, token_ids, img_id_arr, self.s));
        if (audio_token_id > 0) {
            const aud_id_arr = mlx.mlx_array_new_int(@intCast(audio_token_id));
            defer _ = mlx.mlx_array_free(aud_id_arr);
            var aud_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(aud_mask);
            try mlx.check(mlx.mlx_equal(&aud_mask, token_ids, aud_id_arr, self.s));
            var combined = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_logical_or(&combined, mask_2d, aud_mask, self.s));
            _ = mlx.mlx_array_free(mask_2d);
            mask_2d = combined;
        }

        // Expand mask to [B, seq_len, hidden] via broadcast
        const expand_shape = [_]c_int{ h_shape[0], h_shape[1], 1 };
        var mask_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_3d);
        try mlx.check(mlx.mlx_reshape(&mask_3d, mask_2d, &expand_shape, 3, self.s));

        // Broadcast to full shape via logical and with ones
        var mask_expanded = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_expanded);
        {
            var ones_h = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones_h);
            try mlx.check(mlx.mlx_ones(&ones_h, h_shape.ptr, 3, .bool_, self.s));
            try mlx.check(mlx.mlx_multiply(&mask_expanded, mask_3d, ones_h, self.s));
        }

        // Flatten everything
        const total = h_shape[0] * h_shape[1] * h_shape[2];
        const flat_shape = [_]c_int{total};

        var mask_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_flat);
        try mlx.check(mlx.mlx_reshape(&mask_flat, mask_expanded, &flat_shape, 1, self.s));

        // mask_int = mask_flat.astype(int32)
        var mask_int = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_int);
        try mlx.check(mlx.mlx_astype(&mask_int, mask_flat, .int32, self.s));

        // indices = cumsum(mask_int, axis=0) - 1
        var cumsum_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cumsum_arr);
        try mlx.check(mlx.mlx_cumsum(&cumsum_arr, mask_int, 0, false, true, self.s));
        const one_i = mlx.mlx_array_new_int(1);
        defer _ = mlx.mlx_array_free(one_i);
        var indices = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices);
        try mlx.check(mlx.mlx_subtract(&indices, cumsum_arr, one_i, self.s));

        // source = vision_emb.flatten()
        const ve_shape = mlx.getShape(vision_emb);
        const source_size = ve_shape[0] * ve_shape[1] * ve_shape[2];
        const source_shape = [_]c_int{source_size};
        var source_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(source_flat);
        try mlx.check(mlx.mlx_reshape(&source_flat, vision_emb, &source_shape, 1, self.s));

        // indices_mod = indices % source_size
        const source_size_arr = mlx.mlx_array_new_int(source_size);
        defer _ = mlx.mlx_array_free(source_size_arr);
        var indices_mod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices_mod);
        try mlx.check(mlx.mlx_remainder(&indices_mod, indices, source_size_arr, self.s));

        // aligned = source[indices_mod]
        var aligned = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(aligned);
        try mlx.check(mlx.mlx_take(&aligned, source_flat, indices_mod, self.s));

        // Cast aligned to bf16 to match h
        var aligned_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(aligned_bf);
        try mlx.check(mlx.mlx_astype(&aligned_bf, aligned, .bfloat16, self.s));

        // input_flat = h.flatten()
        var input_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input_flat);
        try mlx.check(mlx.mlx_reshape(&input_flat, h, &flat_shape, 1, self.s));

        // result_flat = where(mask_flat, aligned_bf, input_flat)
        var result_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(result_flat);
        try mlx.check(mlx.mlx_where(&result_flat, mask_flat, aligned_bf, input_flat, self.s));

        // Reshape back to [B, seq_len, hidden]
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&result, result_flat, h_shape.ptr, 3, self.s));
        _ = mlx.mlx_array_free(h);
        return result;
    }

    /// Apply vision embeddings to text embeddings during prefill.
    /// Handles scaling and splicing at image_token_id positions.
    /// Returns the (potentially modified) h; caller should replace their h with the result.
    fn applyVisionEmbeddingsWith(self: *Transformer, ctx: *ForwardCtx, h: mlx.mlx_array, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const cfg = &self.config;
        const h_shape = mlx.getShape(h);
        // Only during prefill (seq_len > 1)
        if (h_shape[1] <= 1) return h;
        const ve = ctx.vision_embeddings orelse return h;
        if (cfg.image_token_id == 0) return h;

        // Vision embeddings come out of the MultimodalEmbedder already in text-hidden space;
        // mlx-vlm does NOT re-scale them by sqrt(hidden) the way text embeddings are scaled
        // at LM embedding time. Splice directly — scaling here corrupts the MoE router's
        // magnitude assumptions (visible as "please provide an image" responses on 26B MoE).
        return self.spliceVisionEmbeddings(h, token_ids, ve, cfg.image_token_id, cfg.audio_token_id);
    }

    // ── Activation functions ──

    /// GELU approximate: dispatches to compiled (fused kernel) when available.
    fn gelu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_gelu) |compiled| {
            const in_arr = [_]mlx.mlx_array{x};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        return self.geluUncompiled(x);
    }

    /// Raw GELU implementation (8 ops, used as fallback and as compilation source).
    fn geluUncompiled(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, self.three.?, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, self.gelu_inner.?, x3, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
        var scaled_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled_val);
        try mlx.check(mlx.mlx_multiply(&scaled_val, self.gelu_coeff.?, sum, self.s));
        var tanh_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tanh_val);
        try mlx.check(mlx.mlx_tanh(&tanh_val, scaled_val, self.s));
        var one_plus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_plus);
        try mlx.check(mlx.mlx_add(&one_plus, self.one, tanh_val, self.s));
        var x_times = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_times);
        try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x_times, self.half, self.s));
        return result;
    }

    fn silu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
        return result;
    }

    inline fn mlpActivation(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        return switch (self.config.hidden_act) {
            .gelu_approx => self.gelu(x),
            .silu => self.silu(x),
            .relu_sq => self.reluSquared(x),
        };
    }

    fn reluSquared(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        var relu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(relu);
        try mlx.check(mlx.mlx_maximum(&relu, x, zero, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_square(&result, relu, self.s));
        return result;
    }

    // ── Conv1d with cache (shared by GatedDeltaNet, LFM2, Mamba2) ──

    /// Prepends cached conv state, applies depthwise conv1d, updates cache.
    /// If apply_silu is true, applies SiLU activation after conv.
    /// conv_b is an optional bias added after conv1d.
    fn conv1dWithCache(
        self: *Transformer,
        x: mlx.mlx_array,
        conv_w: mlx.mlx_array,
        conv_b: ?mlx.mlx_array,
        ssm: *SSMCacheEntry,
        batch: c_int,
        cdim: c_int,
        kernel: c_int,
        apply_silu: bool,
    ) !mlx.mlx_array {
        // Prepend conv_state or zeros
        var conv_input: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(conv_input);
        if (ssm.initialized) {
            const arr = [_]mlx.mlx_array{ ssm.conv_state, x };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
        } else {
            const zero_shape = [_]c_int{ batch, kernel - 1, cdim };
            var zero_state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zero_state);
            try mlx.check(mlx.mlx_zeros(&zero_state, &zero_shape, 3, .bfloat16, self.s));
            const arr = [_]mlx.mlx_array{ zero_state, x };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
        }

        // Update conv_state: keep last (kernel-1) positions
        {
            const ci_shape = mlx.getShape(conv_input);
            const ci_len = ci_shape[1];
            const keep_start = ci_len - (kernel - 1);
            const start = [_]c_int{ 0, keep_start, 0 };
            const stop = [_]c_int{ batch, ci_len, cdim };
            const strides = [_]c_int{ 1, 1, 1 };
            var new_conv_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &start, 3, &stop, 3, &strides, 3, self.s));
            _ = mlx.mlx_array_free(ssm.conv_state);
            ssm.conv_state = new_conv_state;
            ssm.initialized = true;
        }

        // PLD capture: stash the full conv input ([B, (kernel-1)+T, conv_dim])
        // so partial-accept rollback can slice the accepted-position conv_state
        // without a re-forward. Refcount-shared; freed at the end of the round.
        if (self.spec_capture_ssm) {
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&ssm.spec_conv_input, conv_input);
        }

        // Depthwise conv1d (groups = cdim)
        var conv_out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&conv_out, conv_input, conv_w, 1, 0, 1, cdim, self.s));

        // Optional bias
        if (conv_b) |cb| {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, conv_out, cb, self.s));
            _ = mlx.mlx_array_free(conv_out);
            conv_out = biased;
        }

        // Optional SiLU activation
        if (apply_silu) {
            const activated = try self.silu(conv_out);
            _ = mlx.mlx_array_free(conv_out);
            return activated;
        }
        return conv_out;
    }

    /// Fused GeGLU: gelu(gate) * up in a single compiled kernel.
    /// Falls back to separate ops if not compiled.
    fn computeGeglu(self: *const Transformer, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_geglu) |compiled| {
            const in_arr = [_]mlx.mlx_array{ gate, up };
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 2);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        // Fallback: separate gelu + multiply
        const activated = try self.mlpActivation(gate);
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, activated, up, self.s));
        return result;
    }

    /// Fused logit softcap: tanh(x/cap) * cap in a single compiled kernel.
    fn applySoftcap(self: *const Transformer, logits: mlx.mlx_array) !mlx.mlx_array {
        if (self.compiled_softcap) |compiled| {
            const in_arr = [_]mlx.mlx_array{logits};
            const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
            defer _ = mlx.mlx_vector_array_free(in_vec);
            var out_vec = mlx.mlx_vector_array{ .ctx = null };
            try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
            defer _ = mlx.mlx_vector_array_free(out_vec);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&result, out_vec, 0));
            return result;
        }
        // Fallback: separate ops
        const cap = self.softcap_scalar.?;
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, logits, cap, self.s));
        defer _ = mlx.mlx_array_free(scaled);
        var tanh_val = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tanh(&tanh_val, scaled, self.s));
        defer _ = mlx.mlx_array_free(tanh_val);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, tanh_val, cap, self.s));
        return result;
    }

    /// SwiGLU: silu(gate) * x
    fn swiglu(self: *const Transformer, gate: mlx.mlx_array, x: mlx.mlx_array) !mlx.mlx_array {
        const activated = try self.silu(gate);
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, activated, x, self.s));
        return result;
    }

    // ── Forward dispatch ──

    const EVAL_EVERY_N_LAYERS: u32 = 48;
    const MOE_EVAL_EVERY_N_LAYERS: u32 = 4;
    const RECURRENCE_EVAL_INTERVAL: usize = 32;

    /// Per-layer prefill transient (bytes) above which the layer loop eval()s
    /// after EVERY layer instead of the coarse default cadence. Between eval
    /// points ~3 layers' transients coexist under MLX lazy eval, so a
    /// multi-GB transient triples; eval-per-layer bounds peak to ~1 layer.
    /// Two transients count (both scale with total_kv, so short prompts keep
    /// the coarse cadence and pay nothing):
    ///   - the composed-SDPA score tensor [heads, chunk, total_kv] — only for
    ///     head_dims no fused kernel covers (prefillHeadDimFused: <= 128 via
    ///     MLX, 256 via msv_attn_p256; unfused only via the kill switch or an
    ///     exotic dim);
    ///   - the dense-fp16 rebuild of a quantized KV cache (denseView runs
    ///     per layer under --kv-quant, over the FULL cache).
    /// Budget calibrated LIVE on gemma-4-26B-A4B-qat-4bit (M4 Max):
    /// eval-per-layer costs ~4.5% prefill at a 5K prompt (845 MB scores —
    /// below this budget, coarse cadence kept) and ~0% at 102K (3.4 GB
    /// scores — flips to per-layer, peak 51.6 -> 27.0 GB, prefill +14%).
    /// 2 GiB puts the flip at ~8K tokens on 16-head/256-hd geometry, where
    /// the ~3% sync cost buys a 13.7 -> 4.3 GB transient bound — the margin
    /// that keeps a 32 GB Mac serving a 26B alive.
    const PREFILL_EVAL_TRANSIENT_BUDGET: u64 = 2 << 30;

    /// Forwards narrower than this skip the mid-loop eval cadence entirely.
    /// Spec-decode VERIFY forwards (PLD/drafter/MTP: seq 2..~9) ride the
    /// prefill layer loops because they're multi-token, but their lazy-graph
    /// transients are decode-scale (KB-MB) — the periodic eval() that bounds
    /// GB-scale prefill-chunk transients only costs them synchronous
    /// pipeline drains. Measured (qwen3.6-27B, 64 layers, cadence 4 = 16
    /// drains per MTP round): depth-1 rounds 48 ms where the AR forward is
    /// ~34 ms, and the drain tax grows superlinearly with draft depth.
    /// seq-1 decode never ran cadence evals, so exempting verify-width
    /// forwards bounds nothing that wasn't already unbounded at decode.
    const PREFILL_EVAL_MIN_SEQ: c_int = 32;

    /// Pure gate: does the mid-loop eval cadence apply to a forward of this
    /// width? (True = real prefill chunk; false = decode/spec-verify shape.)
    pub fn prefillEvalCadenceApplies(seq_len: c_int) bool {
        return seq_len >= PREFILL_EVAL_MIN_SEQ;
    }

    /// Pure cadence pick for the prefill layer loops (standard/MoE/hybrid).
    fn prefillEvalCadence(
        default_cadence: u32,
        head_dim: u32,
        n_heads: u32,
        kv_heads: u32,
        chunk_len: u64,
        total_kv: u64,
        kv_dequant: bool,
    ) u32 {
        const scores: u64 = if (!prefillHeadDimFused(head_dim)) @as(u64, n_heads) * chunk_len * total_kv * 2 else 0;
        const dequant: u64 = if (kv_dequant) 2 * total_kv * @as(u64, kv_heads) * @as(u64, head_dim) * 2 else 0;
        return if (scores + dequant > PREFILL_EVAL_TRANSIENT_BUDGET) 1 else default_cadence;
    }

    /// Default forward context, routing through the Transformer's own state.
    /// Used by the single-slot legacy path and by Phase-2 prefill on a slot
    /// that has had its KVCache temporarily swapped onto the Transformer.
    pub fn defaultCtx(self: *Transformer) ForwardCtx {
        return .{
            .cache = &self.cache,
            .moe_seq_offset = &self.moe_seq_offset,
            .ssm_entries = self.ssm_entries,
            .capture_hidden = self.capture_hidden,
            .vision_embeddings = self.vision_embeddings,
        };
    }

    pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        var ctx = self.defaultCtx();
        return self.forwardWith(&ctx, token_ids);
    }

    pub fn forwardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (self.bert_layers != null) return self.forwardBertWith(ctx, token_ids);
        // Bidirectional embedding models (EmbeddingGemma) load standard gemma3
        // weights but never run causal decode.
        if (self.config.use_bidirectional_attention) return self.forwardGemma3EncoderWith(ctx, token_ids);
        if (self.hybrid_layers != null) return self.forwardHybridWith(ctx, token_ids);
        if (self.moe_layers != null) return self.forwardMoeWith(ctx, token_ids);
        return self.forwardStandardWith(ctx, token_ids);
    }

    /// Free compiled JIT closures (compiled_forward / compiled_gelu /
    /// compiled_geglu / compiled_softcap / compiled_moe_routing). They get
    /// bound to the calling thread's mlx GPU stream at compile time; once
    /// inference moves to a different thread (Phase 2 scheduler) calls
    /// against them fail with "no Stream(gpu, N) in current thread". Clear
    /// them here so subsequent forward calls take the unfused fallback path,
    /// then optionally re-warm on the new thread to recompile against its
    /// own stream.
    pub fn clearCompiledClosures(self: *Transformer) void {
        if (self.compiled_forward) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_forward = null;
        }
        if (self.compiled_gelu) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_gelu = null;
        }
        if (self.compiled_geglu) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_geglu = null;
        }
        if (self.compiled_softcap) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_softcap = null;
        }
        if (self.compiled_moe_routing) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_moe_routing = null;
        }
        if (self.compiled_hy3_routing) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_hy3_routing = null;
        }
        if (self.compiled_gdn_gate) |c| {
            _ = mlx.mlx_closure_free(c);
            self.compiled_gdn_gate = null;
        }
    }

    /// Pre-fault weight pages and trigger first-touch kernel compiles before
    /// the first real request so cold prefill doesn't pay 800+ms of GPU page
    /// faulting (measured on Gemma 4 E4B 4-bit). Runs three forward passes:
    ///   1. [1, 1] decode-shape: faults embed matrix + compiles decode kernel
    ///   2. [1, 8] prefill-shape: compiles short-prefill kernel
    /// then resets the cache so the first real request starts from clean state.
    /// Idempotent — calling twice is wasted work but not incorrect.
    pub fn warmup(self: *Transformer) !void {
        const dummy_id: i32 = 0; // BOS-ish placeholder; the actual id doesn't matter for warmup
        const decode_shape = [_]c_int{ 1, 1 };
        const decode_input = mlx.mlx_array_new_data(&dummy_id, &decode_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(decode_input);
        const decode_logits = try self.forward(decode_input);
        _ = mlx.mlx_array_free(decode_logits);
        // Materialize the cache update so subsequent forwards see initialized entries.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();

        // Reset before the prefill-shape pass so we exercise the cold-init path,
        // not the partial-cache path.
        try self.resetCache();

        const ids_8 = [_]i32{ 0, 0, 0, 0, 0, 0, 0, 0 };
        const prefill_shape = [_]c_int{ 1, 8 };
        const prefill_input = mlx.mlx_array_new_data(&ids_8, &prefill_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(prefill_input);
        const prefill_logits = try self.forward(prefill_input);
        _ = mlx.mlx_array_free(prefill_logits);
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (self.cache.entries) |*entry| {
                if (!entry.initialized) continue;
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
            }
            _ = mlx.mlx_eval(eval_vec);
        }
        _ = mlx.mlx_clear_cache();
        try self.resetCache();
    }

    /// Run a forward pass and ALSO capture the post-final-norm hidden state
    /// at the LAST position into `*out_hidden`. Used by PLD verify-fusion
    /// (which re-uses the captured hidden as part of partial-accept rollback)
    /// and by the Gemma 4 assistant drafter (which needs `h_prev` as a seed
    /// for the next drafter step). Caller owns the captured array (must
    /// `mlx_array_free`). Both `forwardStandard` and `forwardMoe` honor the
    /// capture; other families fall through to a regular forward and leave
    /// `*out_hidden` as a default `mlx_array_new()`.
    pub fn forwardCaptureHidden(
        self: *Transformer,
        token_ids: mlx.mlx_array,
        out_hidden: *mlx.mlx_array,
    ) !mlx.mlx_array {
        std.debug.assert(self.capture_hidden == null); // re-entrant call
        var ctx = self.defaultCtx();
        ctx.capture_hidden = out_hidden;
        return self.forwardWith(&ctx, token_ids);
    }

    /// Variant of `forwardWith` that overrides `ctx.capture_hidden` for this
    /// call only (saved and restored on exit). Used by per-slot generators
    /// (Phase 2) so the capture target is request-local without mutating
    /// shared state on the ctx.
    pub fn forwardWithCapture(
        self: *Transformer,
        ctx: *ForwardCtx,
        token_ids: mlx.mlx_array,
        out_hidden: *mlx.mlx_array,
    ) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        ctx.capture_hidden = out_hidden;
        defer ctx.capture_hidden = saved;
        return self.forwardWith(ctx, token_ids);
    }

    /// Like `forwardWithCapture` but ALSO captures the full `[B, L, H]`
    /// post-final-norm hidden (all positions) into `out_hidden_all`. The MTP
    /// verify forward needs the last-position hidden (next round's h_prev)
    /// and every position's hidden (committed-history re-append) in one pass.
    pub fn forwardWithCaptureAll(
        self: *Transformer,
        ctx: *ForwardCtx,
        token_ids: mlx.mlx_array,
        out_hidden: *mlx.mlx_array,
        out_hidden_all: *mlx.mlx_array,
    ) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        const saved_all = ctx.capture_hidden_all;
        ctx.capture_hidden = out_hidden;
        ctx.capture_hidden_all = out_hidden_all;
        defer {
            ctx.capture_hidden = saved;
            ctx.capture_hidden_all = saved_all;
        }
        return self.forwardWith(ctx, token_ids);
    }

    // ── BERT encoder-only forward pass ──

    fn bertEmbedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        const flat_shape = [_]c_int{batch * seq_len};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        // Word embeddings
        const word_emb = try self.dequantTake(self.emb_w, self.emb_s, self.emb_b, flat_ids);
        defer _ = mlx.mlx_array_free(word_emb);

        // Reshape word embeddings to [B, S, H] up front: position / token-type
        // embeddings are computed once per POSITION ([1, S, H]) and broadcast
        // across the batch — adding them to the flat [B*S, H] form only
        // happened to work at batch == 1.
        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var word_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(word_3d);
        try mlx.check(mlx.mlx_reshape(&word_3d, word_emb, &out_shape, 3, self.s));

        // Position IDs: [0, 1, 2, ..., seq_len-1]
        var pos_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_ids);
        try mlx.check(mlx.mlx_arange(&pos_ids, 0, @as(f64, @floatFromInt(seq_len)), 1, .int32, self.s));

        const pos_emb = try self.dequantTake(self.bert_pos_w, self.bert_pos_s, self.bert_pos_b, pos_ids);
        defer _ = mlx.mlx_array_free(pos_emb);

        // Token type IDs: all zeros (one row per position, broadcast over B)
        const seq_shape = [_]c_int{seq_len};
        var toktype_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(toktype_ids);
        try mlx.check(mlx.mlx_zeros(&toktype_ids, &seq_shape, 1, .int32, self.s));

        const toktype_emb = try self.dequantTake(self.bert_toktype_w, self.bert_toktype_s, self.bert_toktype_b, toktype_ids);
        defer _ = mlx.mlx_array_free(toktype_emb);

        const bcast_shape = [_]c_int{ 1, seq_len, @intCast(self.config.hidden_size) };
        var pos_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_3d);
        try mlx.check(mlx.mlx_reshape(&pos_3d, pos_emb, &bcast_shape, 3, self.s));
        var toktype_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(toktype_3d);
        try mlx.check(mlx.mlx_reshape(&toktype_3d, toktype_emb, &bcast_shape, 3, self.s));

        // Sum: word + position + token_type → [B, S, H]
        var wp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wp);
        try mlx.check(mlx.mlx_add(&wp, word_3d, pos_3d, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, wp, toktype_3d, self.s));

        // LayerNorm
        return self.layerNorm(sum, self.bert_emb_norm_w, self.bert_emb_norm_b);
    }

    fn dequantTake(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !mlx.mlx_array {
        var tw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tw);
        try mlx.check(mlx.mlx_take_axis(&tw, w, ids, 0, self.s));
        if (sc.ctx == null) {
            // Dense bf16: gathered rows are the embeddings; no dequantize.
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&result, tw, .bfloat16, self.s));
            return result;
        }
        var ts = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ts);
        try mlx.check(mlx.mlx_take_axis(&ts, sc, ids, 0, self.s));
        // Bias-less modes ship a null-ctx bi — gather only when present.
        var tb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tb);
        if (bi.ctx != null) {
            try mlx.check(mlx.mlx_take_axis(&tb, bi, ids, 0, self.s));
        }
        var result = mlx.mlx_array_new();
        const qp = self.quantParamsFor(w, sc);
        try mlx.check(mlx.mlx_dequantize(
            &result,
            tw,
            ts,
            tb,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            .{}, // global_scale (null)
            .{ .value = .bfloat16, .has_value = true },
            self.s,
        ));
        return result;
    }

    fn forwardBertWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        // BERT is encoder-only: no per-request KV state; ctx only carries the
        // optional key-padding mask for padded [B, T] embedding batches.
        const pad_mask = ctx.key_pad_mask;
        const bert_layers = self.bert_layers.?;
        const h_count = self.config.num_attention_heads;
        const head_dim = self.config.head_dim;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        var h = try self.bertEmbedding(token_ids);

        for (bert_layers) |lw| {
            // Self-attention
            const q = try self.qmatmulAddBias(h, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
            defer _ = mlx.mlx_array_free(q);
            const k = try self.qmatmulAddBias(h, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
            defer _ = mlx.mlx_array_free(k);
            const v = try self.qmatmulAddBias(h, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
            defer _ = mlx.mlx_array_free(v);

            // Reshape [B, S, H] -> [B, S, heads, head_dim] -> [B, heads, S, head_dim]
            const qkv_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(head_dim) };
            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, &qkv_shape, 4, self.s));
            var k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_r);
            try mlx.check(mlx.mlx_reshape(&k_r, k, &qkv_shape, 4, self.s));
            var v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_r);
            try mlx.check(mlx.mlx_reshape(&v_r, v, &qkv_shape, 4, self.s));

            const perm = [_]c_int{ 0, 2, 1, 3 };
            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_r, &perm, 4, self.s));
            var k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_t);
            try mlx.check(mlx.mlx_transpose_axes(&k_t, k_r, &perm, 4, self.s));
            var v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_t);
            try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

            // Bidirectional attention (no causal mask); padded batches mask
            // their pad KEYS so real positions never attend into padding.
            var attn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
                &attn,
                q_t,
                k_t,
                v_t,
                scale,
                if (pad_mask != null) "array" else "",
                if (pad_mask) |m| m else mlx.mlx_array_new(),
                mlx.mlx_array_new(),
                self.s,
            ));

            // Transpose back [B, heads, S, head_dim] -> [B, S, heads, head_dim] -> [B, S, H]
            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn, &perm, 4, self.s));
            const flat_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

            // Output projection
            const o = try self.qmatmulAddBias(attn_flat, lw.o_w, lw.o_s, lw.o_b, lw.o_bias);
            defer _ = mlx.mlx_array_free(o);

            // Residual + LayerNorm
            var h_plus_attn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(h_plus_attn);
            try mlx.check(mlx.mlx_add(&h_plus_attn, h, o, self.s));
            _ = mlx.mlx_array_free(h);

            h = try self.layerNorm(h_plus_attn, lw.attn_norm_w, lw.attn_norm_b);

            // MLP: intermediate (GELU) -> output
            const inter = try self.qmatmulAddBias(h, lw.inter_w, lw.inter_s, lw.inter_b, lw.inter_bias);
            defer _ = mlx.mlx_array_free(inter);
            const activated = try self.gelu(inter);
            defer _ = mlx.mlx_array_free(activated);
            const out = try self.qmatmulAddBias(activated, lw.out_w, lw.out_s, lw.out_b, lw.out_bias);
            defer _ = mlx.mlx_array_free(out);

            // Residual + LayerNorm
            var h_plus_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(h_plus_out);
            try mlx.check(mlx.mlx_add(&h_plus_out, h, out, self.s));
            _ = mlx.mlx_array_free(h);

            h = try self.layerNorm(h_plus_out, lw.out_norm_w, lw.out_norm_b);
        }

        return h; // [B, S, H] — hidden states for mean pooling
    }

    // ── Gemma 4: Per-Layer Embeddings (PLE) ──

    /// Compute PLE input once before the layer loop.
    /// Returns [B, S, num_layers, ple_dim] combining projected main embeddings + per-layer embeddings.
    fn computePLEInput(self: *Transformer, token_ids: mlx.mlx_array, h: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
        const cfg = &self.config;
        const ple_dim: c_int = @intCast(cfg.hidden_size_per_layer_input);
        const n_layers: c_int = @intCast(cfg.num_hidden_layers);
        const total_ple = n_layers * ple_dim;

        // 1. Per-layer embedding lookup: embed_tokens_per_layer[token_ids] -> [B*S, total_ple]
        const id_shape = mlx.getShape(token_ids);
        const flat_count = id_shape[0] * id_shape[1];
        const flat_shape = [_]c_int{flat_count};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        const ple_emb_raw = try self.dequantTake(self.ple_emb_w, self.ple_emb_s, self.ple_emb_b, flat_ids);
        defer _ = mlx.mlx_array_free(ple_emb_raw);

        // Reshape to [B, S, n_layers, ple_dim] and scale by sqrt(ple_dim)
        const ple_4d_shape = [_]c_int{ batch, seq_len, n_layers, ple_dim };
        var ple_emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ple_emb);
        try mlx.check(mlx.mlx_reshape(&ple_emb, ple_emb_raw, &ple_4d_shape, 4, self.s));

        const emb_scale = bf16Scalar(@sqrt(@as(f32, @floatFromInt(cfg.hidden_size_per_layer_input))), self.s);
        defer _ = mlx.mlx_array_free(emb_scale);
        var ple_emb_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ple_emb_scaled);
        try mlx.check(mlx.mlx_multiply(&ple_emb_scaled, ple_emb, emb_scale, self.s));

        // 2. Project main embeddings: h -> [B, S, total_ple]
        const proj_raw = if (self.ple_proj_quantized)
            try self.qmatmul(h, self.ple_proj_w, self.ple_proj_s, self.ple_proj_b)
        else blk: {
            // Unquantized: regular matmul with transposed weight
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, self.ple_proj_w, self.s));
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&result, h, wt, self.s));
            break :blk result;
        };
        defer _ = mlx.mlx_array_free(proj_raw);

        // Scale by hidden_size^-0.5
        const proj_scale = bf16Scalar(1.0 / @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))), self.s);
        defer _ = mlx.mlx_array_free(proj_scale);
        var proj_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(proj_scaled);
        try mlx.check(mlx.mlx_multiply(&proj_scaled, proj_raw, proj_scale, self.s));

        // Reshape to [B, S, n_layers, ple_dim]
        var proj_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(proj_4d);
        try mlx.check(mlx.mlx_reshape(&proj_4d, proj_scaled, &ple_4d_shape, 4, self.s));

        // RMS norm on last dim (ple_dim)
        var proj_normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(proj_normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&proj_normed, proj_4d, self.ple_proj_norm, cfg.rms_norm_eps, self.s));

        // 3. Combine: (proj_normed + ple_emb_scaled) * (1/sqrt(2))
        var combined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(combined);
        try mlx.check(mlx.mlx_add(&combined, proj_normed, ple_emb_scaled, self.s));

        const inv_sqrt2 = bf16Scalar(1.0 / @sqrt(2.0), self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, combined, inv_sqrt2, self.s));

        _ = &total_ple;
        return result; // [B, S, n_layers, ple_dim]
    }

    /// Apply PLE gating and projection for one layer, modifying h in-place.
    fn applyPLE(self: *Transformer, h_in: mlx.mlx_array, lw: *const LayerWeights, ple_input: mlx.mlx_array, layer_idx: u32, batch: c_int, seq_len: c_int) !mlx.mlx_array {
        const cfg = &self.config;
        const ple_dim: c_int = @intCast(cfg.hidden_size_per_layer_input);

        // Slice ple_input[:, :, layer_idx, :] -> [B, S, ple_dim]
        const li_c: c_int = @intCast(layer_idx);
        const slice_start = [_]c_int{ 0, 0, li_c, 0 };
        const slice_stop = [_]c_int{ batch, seq_len, li_c + 1, ple_dim };
        const slice_strides = [_]c_int{ 1, 1, 1, 1 };
        var ple_slice = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ple_slice);
        try mlx.check(mlx.mlx_slice(&ple_slice, ple_input, &slice_start, 4, &slice_stop, 4, &slice_strides, 4, self.s));

        // Reshape to [B, S, ple_dim]
        const ple_3d_shape = [_]c_int{ batch, seq_len, ple_dim };
        var ple_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ple_3d);
        try mlx.check(mlx.mlx_reshape(&ple_3d, ple_slice, &ple_3d_shape, 3, self.s));

        // gate = gelu(per_layer_input_gate(h))
        const gate_raw = try self.qmatmul(h_in, lw.ple_gate_w.?, lw.ple_gate_s.?, lw.ple_gate_b.?);
        defer _ = mlx.mlx_array_free(gate_raw);
        const gate = try self.gelu(gate_raw);
        defer _ = mlx.mlx_array_free(gate);

        // gated = gate * ple_slice
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, gate, ple_3d, self.s));

        // projected = per_layer_projection(gated) -> [B, S, hidden_size]
        const projected = try self.qmatmul(gated, lw.ple_proj_w.?, lw.ple_proj_s.?, lw.ple_proj_b.?);
        defer _ = mlx.mlx_array_free(projected);

        // normed = rms_norm(projected)
        const normed = try self.rmsNorm(projected, lw.ple_norm.?);
        defer _ = mlx.mlx_array_free(normed);

        // h = h + normed
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&result, h_in, normed, self.s));
        _ = mlx.mlx_array_free(h_in);
        return result;
    }

    // ── Standard forward pass (Gemma / Llama / Qwen3 / Gemma4) ──

    fn forwardStandardWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const offset = ctx.cache.step;
        const cfg = &self.config;
        const h_count = cfg.num_attention_heads;
        const kv_h = cfg.num_key_value_heads;
        const hd = cfg.head_dim;
        const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != hd;
        // Gemma 4: scale = 1.0 because QK-norm handles normalization
        // Gemma 3 and others: 1/sqrt(query_pre_attn_scalar)
        const attn_scale: f32 = if (std.mem.eql(u8, cfg.model_type, "gemma4"))
            1.0
        else
            1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));

        var h = try self.embedding(token_ids);

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;

        // Shapes for sliding-window layers (default)
        const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
        const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };
        // Shapes for global/full-attention layers (only if dual head dims)
        const ghd: u32 = if (has_dual_hd) cfg.global_head_dim else hd;
        const gkv_h: u32 = if (cfg.num_global_key_value_heads > 0) cfg.num_global_key_value_heads else kv_h;
        const q_shape_g = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(ghd) };
        const kv_shape_g = [_]c_int{ batch, seq_len, @intCast(gkv_h), @intCast(ghd) };
        const out_shape_g = [_]c_int{ batch, seq_len, @intCast(h_count * ghd) };

        const perm = [_]c_int{ 0, 2, 1, 3 };
        const perm_back = [_]c_int{ 0, 2, 1, 3 };

        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
        var local_prefill_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_prefill_mask);
        var local_decode_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_decode_mask);

        if (cfg.has_sliding_window) {
            const sw: c_int = @intCast(cfg.sliding_window);
            if (is_prefill) {
                // During prefill, K has all total_kv entries (no windowing in views).
                // The sliding window mask limits attention scope. Skipped when
                // the fused hd-256 kernel band-masks in-kernel — the
                // [1,1,chunk,total_kv] mask is itself GBs at long context; the
                // "array" arm lazily builds it if a per-call check declines.
                if (!(fused256Enabled() and cfg.head_dim == 256 and seq_len >= 2)) {
                    local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, sw);
                }
            }
            if (!is_prefill and total_kv > sw) {
                // During decode, K view has min(total_kv, sw) entries.
                const local_kv_len: c_int = @min(total_kv, sw);
                local_decode_mask = try self.createSlidingWindowDecodeMask(local_kv_len, sw);
            }
        }

        // Eval cadence: drop to per-layer when this chunk's score/dequant
        // transients are large (unfused head_dim > 128 at long ctx, or a
        // quantized cache's dense rebuild) — see prefillEvalCadence.
        const std_eval_cadence = prefillEvalCadence(
            EVAL_EVERY_N_LAYERS,
            @max(hd, ghd),
            h_count,
            @max(kv_h, gkv_h),
            @intCast(seq_len),
            @intCast(total_kv),
            ctx.cache.config.scheme != .off,
        );

        // Gemma 4 PLE: compute per-layer input embeddings once before the layer loop.
        // For vision: zero out image token IDs before PLE (reference: text_mask = ~image_mask).
        var ple_input: ?mlx.mlx_array = null;
        defer {
            if (ple_input) |p| _ = mlx.mlx_array_free(p);
        }
        if (cfg.hidden_size_per_layer_input > 0) {
            if (ctx.vision_embeddings != null and cfg.image_token_id > 0) {
                // Zero out image tokens: per_layer_inputs_tokens = where(text_mask, ids, zeros)
                const img_id = mlx.mlx_array_new_int(@intCast(cfg.image_token_id));
                defer _ = mlx.mlx_array_free(img_id);
                var img_mask = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(img_mask);
                try mlx.check(mlx.mlx_equal(&img_mask, token_ids, img_id, self.s));
                // text_mask = NOT image_mask (invert: 1 where text, 0 where image)
                var text_mask_int = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(text_mask_int);
                try mlx.check(mlx.mlx_astype(&text_mask_int, img_mask, .int32, self.s));
                var ones_int = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ones_int);
                const ones_s = [_]c_int{ batch, seq_len };
                try mlx.check(mlx.mlx_ones(&ones_int, &ones_s, 2, .int32, self.s));
                var text_mask = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(text_mask);
                try mlx.check(mlx.mlx_subtract(&text_mask, ones_int, text_mask_int, self.s));
                // ple_ids = token_ids * text_mask (zeros at image positions)
                var ple_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ple_ids);
                try mlx.check(mlx.mlx_multiply(&ple_ids, token_ids, text_mask, self.s));
                ple_input = try self.computePLEInput(ple_ids, h, batch, seq_len);
            } else {
                ple_input = try self.computePLEInput(token_ids, h, batch, seq_len);
            }
        }

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &self.layers[layer_idx];
            const is_global = cfg.isGlobalLayer(li);
            const is_kv_shared = lw.kv_source != null;

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Pick shapes based on layer type
            const cur_q_shape: *const [4]c_int = if (has_dual_hd and is_global) &q_shape_g else &q_shape;
            const cur_kv_shape: *const [4]c_int = if (has_dual_hd and is_global) &kv_shape_g else &kv_shape;
            const cur_out_shape: *const [3]c_int = if (has_dual_hd and is_global) &out_shape_g else &out_shape;
            const cur_hd: u32 = if (has_dual_hd and is_global) ghd else hd;
            // RoPE dims: for global layers with partial rotary, only rotate part of head_dim
            const rope_dims: c_int = if (is_global and cfg.partial_rotary_factor_global < 1.0)
                @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(cur_hd)) * cfg.partial_rotary_factor_global)))
            else
                @intCast(cur_hd);

            // Q projection
            const q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
            defer _ = mlx.mlx_array_free(q);

            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, cur_q_shape, 4, self.s));

            // Q norm
            const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
            defer {
                if (q_normed) |qn| _ = mlx.mlx_array_free(qn);
            }
            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));

            // RoPE on Q (proportional for global layers when available)
            const use_prop_rope = is_global and self.rope_freqs_global != null;
            const rope_base_opt = mlx.mlx_optional_float{
                .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
                .has_value = !use_prop_rope,
            };
            const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
            const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
            // When using proportional RoPE, pass full head_dim (freqs handle partial rotation via inf padding)
            const effective_rope_dims: c_int = if (use_prop_rope) @intCast(cur_hd) else rope_dims;
            var q_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_rope);
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, effective_rope_dims, false, rope_base_opt, rope_scale, @intCast(offset), rope_freqs, self.s));

            // K, V and cache — either compute or read from shared source.
            // `kv_view` is the lifetime owner: in dense mode it aliases the
            // cache's view (no-op deinit); in quant mode it owns dequantized
            // dense arrays freed at scope exit, after SDPA has consumed them.
            var kv_view: DenseKVView = .{ .k = .{}, .v = .{}, .owned = false };
            defer kv_view.deinit();
            var full_k: mlx.mlx_array = undefined;
            var full_v: mlx.mlx_array = undefined;

            if (is_kv_shared) {
                // KV sharing: read from source layer's cache
                const src = lw.kv_source.?;
                kv_view = try ctx.cache.denseView(src, self.s);
                full_k = kv_view.k;
                full_v = kv_view.v;
            } else {
                // Compute K, V (temp arrays scoped to this block).
                // When k_eq_v, V shares the K projection — compute once, alias into V.
                const own_k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
                defer _ = mlx.mlx_array_free(own_k);
                const own_v = if (lw.k_eq_v)
                    own_k
                else
                    try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
                defer if (!lw.k_eq_v) {
                    _ = mlx.mlx_array_free(own_v);
                };

                var own_k_r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_r);
                var own_v_r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_r);
                try mlx.check(mlx.mlx_reshape(&own_k_r, own_k, cur_kv_shape, 4, self.s));
                try mlx.check(mlx.mlx_reshape(&own_v_r, own_v, cur_kv_shape, 4, self.s));

                // K norm
                var own_k_normed_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_normed_arr);
                if (lw.k_norm) |kn| {
                    own_k_normed_arr = try self.rmsNorm(own_k_r, kn);
                }
                const k_for_rope = if (lw.k_norm != null) own_k_normed_arr else own_k_r;

                // V norm (Gemma 4: parameter-free RMS norm on values)
                var own_v_normed_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_normed_arr);
                if (cfg.has_v_norm) {
                    const vnw = if (has_dual_hd and is_global)
                        (self.v_norm_weight_global orelse self.v_norm_weight.?)
                    else
                        self.v_norm_weight.?;
                    own_v_normed_arr = try self.rmsNorm(own_v_r, vnw);
                }
                const v_after_norm = if (cfg.has_v_norm) own_v_normed_arr else own_v_r;

                var own_k_t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_t);
                var own_v_t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_t);
                try mlx.check(mlx.mlx_transpose_axes(&own_k_t, k_for_rope, &perm, 4, self.s));
                try mlx.check(mlx.mlx_transpose_axes(&own_v_t, v_after_norm, &perm, 4, self.s));

                // RoPE on K
                var own_k_rope = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_rope);
                try mlx.check(mlx.mlx_fast_rope(&own_k_rope, own_k_t, effective_rope_dims, false, rope_base_opt, rope_scale, @intCast(offset), rope_freqs, self.s));

                // Update KV cache
                const max_kv: u32 = if (is_global) 0 else if (cfg.has_sliding_window) cfg.sliding_window else 0;
                kv_view = try ctx.cache.update(li, own_k_rope, own_v_t, self.s, max_kv);
                full_k = kv_view.k;
                full_v = kv_view.v;
            }

            // Scaled dot-product attention
            var attn_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_out);

            // Resolve mask first so dense + fused paths share the selection.
            var sel_mode: []const u8 = "";
            var sel_mask: mlx.mlx_array = none_mask;
            if (!cfg.has_sliding_window) {
                if (is_prefill) sel_mode = "causal";
            } else {
                const sw: c_int = @intCast(cfg.sliding_window);
                if (is_prefill and is_global) {
                    sel_mode = "causal";
                } else if (is_prefill) {
                    sel_mode = "array";
                    sel_mask = local_prefill_mask;
                } else if (is_global) {
                    // Global layers: full attention, no mask (defaults).
                } else if (blk: {
                    const check_layer = if (is_kv_shared) lw.kv_source.? else li;
                    break :blk @as(c_int, @intCast(ctx.cache.seqLen(check_layer))) <= sw;
                }) {
                    // Within window: no mask needed (defaults).
                } else {
                    sel_mode = "array";
                    sel_mask = local_decode_mask;
                }
            }

            // Fused-attn opt-in: consume the cache's quant triples directly
            // via mlx_quantized_matmul. Only when the request opts in AND
            // the cache scheme is .affine (TurboQuant variants need their
            // rotation undo step, deferred). Falls back to dense SDPA on
            // any precondition miss.
            if (ctx.kv_attn_fused and kv_view.has_quant_triple) {
                const fused = try kv_quant.quantAttention(
                    q_rope,
                    kv_view.kTriple(),
                    kv_view.vTriple(),
                    kv_view.bits,
                    kv_view.group_size,
                    attn_scale,
                    sel_mode,
                    sel_mask,
                    self.s,
                );
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else if (std.mem.eql(u8, sel_mode, "causal")) {
                if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                    _ = mlx.mlx_array_free(attn_out);
                    attn_out = fused;
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                }
            } else if (std.mem.eql(u8, sel_mode, "array")) {
                var fused_done = false;
                if (is_prefill and cfg.has_sliding_window) {
                    // Sliding-window prefill: the band mask runs in-kernel.
                    if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, @intCast(cfg.sliding_window))) |fused| {
                        _ = mlx.mlx_array_free(attn_out);
                        attn_out = fused;
                        fused_done = true;
                    }
                }
                if (!fused_done) {
                    if (sel_mask.ctx == null) {
                        // The eager mask build was skipped because the fused
                        // kernel was expected to cover this layer but a
                        // per-call precondition declined — build it now (and
                        // keep it for the remaining layers).
                        local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, @intCast(cfg.sliding_window));
                        sel_mask = local_prefill_mask;
                    }
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", sel_mask, .{ .ctx = null }, self.s));
                }
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            }

            // Reshape attention output
            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, cur_out_shape, 3, self.s));

            const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
            defer _ = mlx.mlx_array_free(o_out);

            // MLP with pre/post FF norms (Gemma 3/4 style) or simple residual (Llama style)
            if (cfg.has_pre_ff_norm) {
                const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(attn_normed);
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
                defer _ = mlx.mlx_array_free(ff_normed);

                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                const gate_up = try self.computeGeglu(gate_raw, up);
                defer _ = mlx.mlx_array_free(gate_up);
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                const mlp_normed = try self.rmsNorm(down, lw.post_ff_norm.?);
                defer _ = mlx.mlx_array_free(mlp_normed);
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            } else {
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, o_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(ff_normed);

                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                const gate_up = try self.computeGeglu(gate_raw, up);
                defer _ = mlx.mlx_array_free(gate_up);
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, down, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            // Gemma 4 PLE: apply per-layer embedding gate + projection (AFTER attention+MLP)
            if (ple_input != null and lw.ple_gate_w != null) {
                h = try self.applyPLE(h, lw, ple_input.?, li, batch, seq_len);
            }

            // Gemma 4: layer_scalar
            if (lw.layer_scalar) |ls| {
                var h_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_scaled;
            }

            if (is_prefill and prefillEvalCadenceApplies(seq_len) and (layer_idx + 1) % std_eval_cadence == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);

        // Speculative-decoding capture: slice the LAST position of the
        // post-final-norm hidden into `capture_hidden` (refcount-shared).
        // Used by PLD verify-fusion and the Gemma 4 assistant drafter
        // (which needs the post-final-norm hidden as h_prev seed). Mirrors
        // the identical block in forwardMoe.
        if (ctx.capture_hidden) |target| {
            const fn_shape = mlx.getShape(final_normed);
            const last = fn_shape[1] - 1;
            const start = [_]c_int{ 0, last, 0 };
            const stop = [_]c_int{ fn_shape[0], fn_shape[1], fn_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            var sliced = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&sliced, final_normed, &start, 3, &stop, 3, &strides, 3, self.s));
            _ = mlx.mlx_array_set(target, sliced);
            _ = mlx.mlx_array_free(sliced);
        }
        if (ctx.capture_hidden_all) |target_all| {
            _ = mlx.mlx_array_set(target_all, final_normed);
        }

        if (self.embedding_mode) return final_normed;
        var logits = try self.lmHeadProject(final_normed);
        _ = mlx.mlx_array_free(final_normed);

        // Gemma 4: logit softcapping — tanh(logits / cap) * cap
        if (self.softcap_scalar != null) {
            const capped = try self.applySoftcap(logits);
            _ = mlx.mlx_array_free(logits);
            logits = capped;
        }

        return logits;
    }

    // ── Batched-decode forward pass ──
    //
    // One forward call computes next-token logits for N concurrent requests at
    // decode step (q_len=1 each). Each request owns its own KVCache via its
    // ForwardCtx; weights are shared. Returns N logits arrays of shape [1,1,V],
    // one per slot in the input order. Caller owns the returned slice and the
    // inner arrays (free each via mlx_array_free, then allocator.free the slice).
    //
    // Restrictions (enforced upstream by `Scheduler.batchable`):
    //   - Standard arch only (no MoE, hybrid SSM, encoder-only).
    //   - Decode only (each slot contributes exactly one new token).
    //   - No grammar-constrained sampling, no in-flight speculative round.
    //
    // Per-layer flow:
    //   embed → input_norm → Q/K/V proj (B=N, batch-invariant)
    //   → Q/K-norm → transpose → mlx_fast_rope_dynamic (per-slot offset)
    //   → per-slot cache.update at B=1 (each ctx's cache owns its own state)
    //   → gather views, pad to common kv_max, concat axis=0
    //   → build [N,1,1,kv_max] additive mask via positions < kv_lens
    //   → SDPA → o_proj → MLP → final_norm → lm_head → softcap → demux.
    pub fn forwardBatchedDecode(
        self: *Transformer,
        next_tokens: []const u32,
        ctxs: []const *ForwardCtx,
        rope_offsets: []const u32,
    ) ![]mlx.mlx_array {
        const N: c_int = @intCast(next_tokens.len);
        std.debug.assert(next_tokens.len == ctxs.len);
        std.debug.assert(next_tokens.len == rope_offsets.len);
        std.debug.assert(N >= 1);

        const cfg = &self.config;
        const h_count = cfg.num_attention_heads;
        const kv_h = cfg.num_key_value_heads;
        const hd = cfg.head_dim;
        const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != hd;
        const ghd: u32 = if (has_dual_hd) cfg.global_head_dim else hd;
        const gkv_h: u32 = if (cfg.num_global_key_value_heads > 0) cfg.num_global_key_value_heads else kv_h;
        const attn_scale: f32 = if (std.mem.eql(u8, cfg.model_type, "gemma4"))
            1.0
        else
            1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));

        // 1. Build [N, 1] int32 token tensor from u32 input.
        var token_buf = try self.allocator.alloc(i32, next_tokens.len);
        defer self.allocator.free(token_buf);
        for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
        const tok_shape = [_]c_int{ N, 1 };
        const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(token_arr);

        // 2. Embed → [N, 1, hidden].
        var h = try self.embedding(token_arr);

        // 3. Build per-slot int32 rope-offset array for mlx_fast_rope_dynamic.
        var rope_off_buf = try self.allocator.alloc(i32, rope_offsets.len);
        defer self.allocator.free(rope_off_buf);
        for (rope_offsets, 0..) |o, i| rope_off_buf[i] = @intCast(o);
        const rope_off_shape = [_]c_int{N};
        const rope_offset_arr = mlx.mlx_array_new_data(rope_off_buf.ptr, &rope_off_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(rope_offset_arr);

        // 4. Gemma 4 PLE input — computed once across the batch (token_arr is [N,1]).
        var ple_input: ?mlx.mlx_array = null;
        defer if (ple_input) |p| {
            _ = mlx.mlx_array_free(p);
        };
        if (cfg.hidden_size_per_layer_input > 0) {
            ple_input = try self.computePLEInput(token_arr, h, N, 1);
        }

        const perm = [_]c_int{ 0, 2, 1, 3 };
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Per-slot int32 kv-len buffer reused for the mask each layer.
        var kv_len_buf = try self.allocator.alloc(i32, next_tokens.len);
        defer self.allocator.free(kv_len_buf);

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &self.layers[layer_idx];
            const is_global = cfg.isGlobalLayer(li);
            const is_kv_shared = lw.kv_source != null;

            const cur_hd: u32 = if (has_dual_hd and is_global) ghd else hd;
            const cur_kv_h: u32 = if (has_dual_hd and is_global) gkv_h else kv_h;
            const cur_q_shape = [_]c_int{ N, 1, @intCast(h_count), @intCast(cur_hd) };
            const cur_kv_shape = [_]c_int{ N, 1, @intCast(cur_kv_h), @intCast(cur_hd) };
            const cur_out_shape = [_]c_int{ N, 1, @intCast(h_count * cur_hd) };
            // RoPE dims — same logic as forwardStandard's decode path.
            const use_prop_rope = is_global and self.rope_freqs_global != null;
            const rope_dims_partial: c_int = if (is_global and cfg.partial_rotary_factor_global < 1.0)
                @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(cur_hd)) * cfg.partial_rotary_factor_global)))
            else
                @intCast(cur_hd);
            const rope_base_opt = mlx.mlx_optional_float{
                .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
                .has_value = !use_prop_rope,
            };
            const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
            const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
            const effective_rope_dims: c_int = if (use_prop_rope) @intCast(cur_hd) else rope_dims_partial;

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Q projection + reshape + Q-norm + transpose + dynamic RoPE.
            const q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
            defer _ = mlx.mlx_array_free(q);

            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, &cur_q_shape, 4, self.s));

            const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
            defer if (q_normed) |qn| {
                _ = mlx.mlx_array_free(qn);
            };
            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));

            var q_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_rope);
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, effective_rope_dims, false, rope_base_opt, rope_scale, rope_offset_arr, rope_freqs, self.s));

            const max_kv_per_layer: u32 = if (is_global) 0 else if (cfg.has_sliding_window) cfg.sliding_window else 0;

            // Per-slot KV update (or KV-share lookup), then gather views.
            // own_views holds per-slot [1, kv_h, kv_len_i, cur_hd] mlx_arrays.
            // We pad each to the common kv_max and concat axis=0 → [N, kv_h, kv_max, cur_hd].
            if (!is_kv_shared) {
                // Project K, V at full batch (B=N), reshape, normalize, transpose, RoPE.
                const own_k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
                defer _ = mlx.mlx_array_free(own_k);
                const own_v = if (lw.k_eq_v)
                    own_k
                else
                    try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
                defer if (!lw.k_eq_v) {
                    _ = mlx.mlx_array_free(own_v);
                };

                var own_k_r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_r);
                var own_v_r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_r);
                try mlx.check(mlx.mlx_reshape(&own_k_r, own_k, &cur_kv_shape, 4, self.s));
                try mlx.check(mlx.mlx_reshape(&own_v_r, own_v, &cur_kv_shape, 4, self.s));

                var own_k_normed_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_normed_arr);
                if (lw.k_norm) |kn| {
                    own_k_normed_arr = try self.rmsNorm(own_k_r, kn);
                }
                const k_for_rope = if (lw.k_norm != null) own_k_normed_arr else own_k_r;

                var own_v_normed_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_normed_arr);
                if (cfg.has_v_norm) {
                    const vnw = if (has_dual_hd and is_global)
                        (self.v_norm_weight_global orelse self.v_norm_weight.?)
                    else
                        self.v_norm_weight.?;
                    own_v_normed_arr = try self.rmsNorm(own_v_r, vnw);
                }
                const v_after_norm = if (cfg.has_v_norm) own_v_normed_arr else own_v_r;

                var own_k_t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_t);
                var own_v_t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_v_t);
                try mlx.check(mlx.mlx_transpose_axes(&own_k_t, k_for_rope, &perm, 4, self.s));
                try mlx.check(mlx.mlx_transpose_axes(&own_v_t, v_after_norm, &perm, 4, self.s));

                var own_k_rope = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(own_k_rope);
                try mlx.check(mlx.mlx_fast_rope_dynamic(&own_k_rope, own_k_t, effective_rope_dims, false, rope_base_opt, rope_scale, rope_offset_arr, rope_freqs, self.s));

                // Per-slot cache update at B=1 — slice axis 0 of stacked tensors.
                const k_shape_full = mlx.getShape(own_k_rope);
                const k_h_dim = k_shape_full[1];
                const k_hd_dim = k_shape_full[3];
                for (ctxs, 0..) |slot_ctx, i| {
                    const i_c: c_int = @intCast(i);
                    const slc_start = [_]c_int{ i_c, 0, 0, 0 };
                    const slc_stop = [_]c_int{ i_c + 1, k_h_dim, 1, k_hd_dim };
                    const slc_strides = [_]c_int{ 1, 1, 1, 1 };
                    var k_slot = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(k_slot);
                    var v_slot = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(v_slot);
                    try mlx.check(mlx.mlx_slice(&k_slot, own_k_rope, &slc_start, 4, &slc_stop, 4, &slc_strides, 4, self.s));
                    try mlx.check(mlx.mlx_slice(&v_slot, own_v_t, &slc_start, 4, &slc_stop, 4, &slc_strides, 4, self.s));
                    var slot_view = try slot_ctx.cache.update(li, k_slot, v_slot, self.s, max_kv_per_layer);
                    slot_view.deinit();
                }
            }

            // Gather per-slot views and find kv_max. For KV-shared layers the
            // source layer's view is what we read. Reads go through denseView
            // (kv-quant contract) — the cache's raw key_view/value_view hold
            // packed quantized words under --kv-quant, and feeding those to
            // SDPA is a fatal MLX shape error.
            const view_layer: u32 = if (is_kv_shared) lw.kv_source.? else li;
            const dense_views = try self.allocator.alloc(DenseKVView, ctxs.len);
            defer {
                for (dense_views) |*dv| dv.deinit();
                self.allocator.free(dense_views);
            }
            for (dense_views) |*dv| dv.* = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false };
            var kv_max: c_int = 0;
            for (ctxs, 0..) |slot_ctx, i| {
                dense_views[i] = try slot_ctx.cache.denseView(view_layer, self.s);
                const vshape = mlx.getShape(dense_views[i].k);
                const klen: c_int = vshape[2];
                kv_len_buf[i] = klen;
                if (klen > kv_max) kv_max = klen;
            }

            // Pad every slot view to [1, cur_kv_h, kv_max, cur_hd] and concat axis=0.
            const stacked_k = try self.padAndStackBatchedKV(dense_views, true, kv_max);
            defer _ = mlx.mlx_array_free(stacked_k);
            const stacked_v = try self.padAndStackBatchedKV(dense_views, false, kv_max);
            defer _ = mlx.mlx_array_free(stacked_v);

            // Mask: positions [1,1,1,kv_max] vs kv_lens [N,1,1,1] → broadcast to [N,1,1,kv_max].
            const stacked_mask = try self.buildBatchedDecodeMask(kv_len_buf, kv_max);
            defer _ = mlx.mlx_array_free(stacked_mask);

            // SDPA → [N, h_count, 1, cur_hd].
            var attn_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_out);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, stacked_k, stacked_v, attn_scale, "array", stacked_mask, .{ .ctx = null }, self.s));

            // Output projection.
            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &cur_out_shape, 3, self.s));

            const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
            defer _ = mlx.mlx_array_free(o_out);

            // MLP path — mirrors forwardStandard exactly.
            if (cfg.has_pre_ff_norm) {
                const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(attn_normed);
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
                defer _ = mlx.mlx_array_free(ff_normed);
                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                const gate_up = try self.computeGeglu(gate_raw, up);
                defer _ = mlx.mlx_array_free(gate_up);
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                const mlp_normed = try self.rmsNorm(down, lw.post_ff_norm.?);
                defer _ = mlx.mlx_array_free(mlp_normed);
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            } else {
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, o_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(ff_normed);
                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                const gate_up = try self.computeGeglu(gate_raw, up);
                defer _ = mlx.mlx_array_free(gate_up);
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, down, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            // Gemma 4 PLE: per-layer projection gate.
            if (ple_input != null and lw.ple_gate_w != null) {
                h = try self.applyPLE(h, lw, ple_input.?, li, N, 1);
            }

            // Gemma 4: layer scalar.
            if (lw.layer_scalar) |ls| {
                var h_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_scaled;
            }
        }

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);

        var logits = try self.lmHeadProject(final_normed);
        _ = mlx.mlx_array_free(final_normed);

        if (self.softcap_scalar != null) {
            const capped = try self.applySoftcap(logits);
            _ = mlx.mlx_array_free(logits);
            logits = capped;
        }
        defer _ = mlx.mlx_array_free(logits);

        // Demux: slice axis 0 into N tensors of shape [1, 1, V].
        const lshape = mlx.getShape(logits);
        const vocab: c_int = lshape[2];
        const out = try self.allocator.alloc(mlx.mlx_array, next_tokens.len);
        errdefer self.allocator.free(out);
        for (out, 0..) |*slot, i| {
            const i_c: c_int = @intCast(i);
            const start = [_]c_int{ i_c, 0, 0 };
            const stop = [_]c_int{ i_c + 1, 1, vocab };
            const strides = [_]c_int{ 1, 1, 1 };
            slot.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(slot, logits, &start, 3, &stop, 3, &strides, 3, self.s));
        }
        return out;
    }

    // Pads each slot's dense KV view (shape [1, kv_h, kv_len_i, head_dim]) to a
    // common kv_max along axis 2 with bf16 zeros and concatenates axis 0 →
    // [N, kv_h, kv_max, head_dim]. Views come from KVCache.denseView so quant
    // schemes are already dequantized; `key_not_value` selects k or v.
    fn padAndStackBatchedKV(
        self: *const Transformer,
        views: []const DenseKVView,
        key_not_value: bool,
        kv_max: c_int,
    ) !mlx.mlx_array {
        const pad_axes = [_]c_int{2};
        const pad_value = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(pad_value);

        const padded_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(padded_vec);
        // Track padded arrays so we can free them after the concat.
        var padded_arrs = try self.allocator.alloc(mlx.mlx_array, views.len);
        defer {
            for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(padded_arrs);
        }

        for (views, 0..) |dv, i| {
            const view = if (key_not_value) dv.k else dv.v;
            const vshape = mlx.getShape(view);
            const klen: c_int = vshape[2];
            const high_pad: c_int = kv_max - klen;
            const low_pad_arr = [_]c_int{0};
            const high_pad_arr = [_]c_int{high_pad};
            padded_arrs[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_pad(
                &padded_arrs[i],
                view,
                &pad_axes,
                1,
                &low_pad_arr,
                1,
                &high_pad_arr,
                1,
                pad_value,
                "constant",
                self.s,
            ));
            _ = mlx.mlx_vector_array_append_value(padded_vec, padded_arrs[i]);
        }

        var stacked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&stacked, padded_vec, 0, self.s));
        return stacked;
    }

    // Builds the additive per-slot decode mask [N,1,1,kv_max] in bf16 where
    // valid columns are 0 and out-of-range columns are -inf. Computed via
    // broadcasting: positions[1,1,1,kv_max] < kv_lens[N,1,1,1].
    fn buildBatchedDecodeMask(
        self: *const Transformer,
        kv_lens: []const i32,
        kv_max: c_int,
    ) !mlx.mlx_array {
        const N: c_int = @intCast(kv_lens.len);
        var positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positions);
        try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, self.s));
        const pos_shape = [_]c_int{ 1, 1, 1, kv_max };
        var pos_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_4d);
        try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &pos_shape, 4, self.s));

        const lens_shape = [_]c_int{N};
        const lens_arr = mlx.mlx_array_new_data(kv_lens.ptr, &lens_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(lens_arr);
        const lens_4shape = [_]c_int{ N, 1, 1, 1 };
        var lens_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lens_4d);
        try mlx.check(mlx.mlx_reshape(&lens_4d, lens_arr, &lens_4shape, 4, self.s));

        var valid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(valid);
        try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, self.s));

        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, self.s));
        return mask;
    }

    // ── MoE forward pass (Qwen3.5 + Gemma 4) ──

    fn forwardMoeWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const ml = self.moe_layers.?;
        const offset = ctx.moe_seq_offset.*;
        const cfg = &self.config;
        const is_gemma4 = cfg.isGemma4Layers();
        const is_laguna = std.mem.eql(u8, cfg.model_type, "laguna");

        // PLD spec-decode: thread the per-position SSM capture flag down to the
        // GatedDeltaNet layers (which don't take the ctx). Reset on exit so it
        // never leaks into a non-capturing forward.
        self.spec_capture_ssm = ctx.capture_ssm_seq;
        defer self.spec_capture_ssm = false;

        // Decode sub-block profiler: start the clock before embedding.
        const prof_on = decodeProfileEnabled();
        var pclk: ProfClock = if (prof_on) ProfClock.init() else undefined;

        var h = try self.embedding(token_ids);

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;
        const prof = prof_on and seq_len == 1; // profile decode only
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(h));
            decode_prof.embed_ns += pclk.lap();
        }

        // Qwen3-VL interleaved M-RoPE: build the per-prefill-chunk cos/sin once
        // (shared by every full-attn layer this forward), freed after the loop.
        // Generated-token forwards start past the explicit position table and
        // use the scalar offset+delta path, including multi-token speculative
        // verification.
        if (ctx.mrope_pos != null and is_prefill and @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)) <= ctx.mrope_total) {
            const positions = mrope.PositionContext{
                .pos = ctx.mrope_pos.?,
                .total = ctx.mrope_total,
                .delta = ctx.mrope_delta,
            };
            const cs = try self.buildMropeCosSin(positions, @intCast(offset), @intCast(seq_len));
            ctx.mrope_cos_cur = cs.cos;
            ctx.mrope_sin_cur = cs.sin;
        }
        defer {
            if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
            if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
            ctx.mrope_cos_cur = null;
            ctx.mrope_sin_cur = null;
        }

        // Precompute sliding window masks (Gemma 4 + Laguna sliding layers)
        var local_prefill_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_prefill_mask);
        var local_decode_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_decode_mask);

        if ((is_gemma4 or is_laguna) and cfg.has_sliding_window) {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
            if (is_prefill) {
                // Skipped when the fused hd-256 kernel band-masks in-kernel
                // (the mask itself is chunk x total_kv — GBs at long ctx);
                // gemma4MoeAttnWith lazily builds it if a call declines.
                if (!(fused256Enabled() and cfg.head_dim == 256 and seq_len >= 2)) {
                    local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, sw);
                }
            }
            if (!is_prefill and total_kv > sw) {
                const local_kv_len: c_int = @min(total_kv, sw);
                local_decode_mask = try self.createSlidingWindowDecodeMask(local_kv_len, sw);
            }
        }

        // Eval cadence: drop to per-layer when this chunk's score/dequant
        // transients are large (unfused head_dim > 128 at long ctx, or a
        // quantized cache's dense rebuild) — see prefillEvalCadence.
        const moe_eval_cadence = prefillEvalCadence(
            MOE_EVAL_EVERY_N_LAYERS,
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            @intCast(seq_len),
            @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
            ctx.cache.config.scheme != .off,
        );

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &ml[layer_idx];

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Attention: linear (GatedDeltaNet) or full
            const attn_out = switch (lw.attn) {
                .linear => |la| try self.gatedDeltaNet(normed, &la, &ctx.ssm_entries.?[layer_idx], batch, seq_len),
                .full => |fa| if (is_laguna)
                    try self.lagunaAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask, local_decode_mask)
                else if (is_gemma4)
                    try self.gemma4MoeAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask, local_decode_mask)
                else
                    try self.gatedFullAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill),
            };
            defer _ = mlx.mlx_array_free(attn_out);
            if (prof) {
                try mlx.check(mlx.mlx_array_eval(attn_out));
                decode_prof.attn_ns += pclk.lap();
            }

            if (is_gemma4) {
                h = try self.gemma4MoeLayerTail(h, attn_out, lw, ctx.use_encoder_scalars);
            } else {
                // Qwen3.5: simple residual + post_attn_norm before MLP
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(ff_normed);
                const mlp_out = switch (lw.mlp) {
                    .moe => |*mw| try self.moeMLP(ff_normed, mw),
                    .dense => |*dw| try self.denseMLP(ff_normed, dw),
                };
                defer _ = mlx.mlx_array_free(mlp_out);

                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            if (prof) {
                try mlx.check(mlx.mlx_array_eval(h));
                decode_prof.mlp_ns += pclk.lap();
            }

            if (is_prefill and prefillEvalCadenceApplies(seq_len) and (layer_idx + 1) % moe_eval_cadence == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        ctx.moe_seq_offset.* += @intCast(seq_len);

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);

        // Speculative-decoding capture: slice the post-final-norm hidden
        // at the LAST position only. Used by PLD verify-fusion and the
        // Gemma 4 assistant drafter as `h_prev`. Caller frees the captured
        // array.
        if (ctx.capture_hidden) |target| {
            const fn_shape = mlx.getShape(final_normed);
            const last = fn_shape[1] - 1;
            const start = [_]c_int{ 0, last, 0 };
            const stop = [_]c_int{ fn_shape[0], fn_shape[1], fn_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            var sliced = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&sliced, final_normed, &start, 3, &stop, 3, &strides, 3, self.s));
            _ = mlx.mlx_array_set(target, sliced);
            _ = mlx.mlx_array_free(sliced);
        }
        if (ctx.capture_hidden_all) |target_all| {
            _ = mlx.mlx_array_set(target_all, final_normed);
        }

        if (self.embedding_mode) return final_normed;
        // Diffusion encoder passes exist only to fill the KV cache — the
        // 262K-vocab projection is pure waste there.
        if (ctx.skip_lm_head) return final_normed;
        var logits = try self.lmHeadProject(final_normed);
        _ = mlx.mlx_array_free(final_normed);

        // Gemma 4: logit softcapping — tanh(logits / cap) * cap
        if (self.softcap_scalar != null) {
            const capped = try self.applySoftcap(logits);
            _ = mlx.mlx_array_free(logits);
            logits = capped;
        }

        if (prof) {
            try mlx.check(mlx.mlx_array_eval(logits));
            decode_prof.lmhead_ns += pclk.lap();
            decode_prof.calls += 1;
            if (decode_prof.calls % 64 == 0) decodeProfReport();
        }

        return logits;
    }

    /// Gemma 4 / DiffusionGemma layer tail (from the HF reference):
    ///   h = residual + post_attn_norm(attn_out)          [attn_out borrowed]
    ///   residual = h
    ///   shared  = post_ff_norm_1(mlp(pre_ff_norm(h)))
    ///   experts = post_ff_norm_2(moe(pre_ff_norm_2(residual)))  [router sees RAW residual]
    ///   h = residual + post_ff_norm(shared + experts)
    ///   h *= layer_scalar (encoder variant when requested and bound)
    /// Consumes `h_in`; returns the new layer output (caller owns).
    fn gemma4MoeLayerTail(self: *Transformer, h_in: mlx.mlx_array, attn_out: mlx.mlx_array, lw: *const MoeLayerWeights, use_encoder_scalar: bool) !mlx.mlx_array {
        var h = h_in;

        // Attention residual
        const attn_normed = try self.rmsNorm(attn_out, lw.post_attn_norm);
        defer _ = mlx.mlx_array_free(attn_normed);
        var h_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_new;
        // h is now the residual for the feedforward block

        // Shared expert: pre_ff_norm → mlp → post_ff_norm_1
        const shared_in = try self.rmsNorm(h, lw.pre_ff_norm.?);
        defer _ = mlx.mlx_array_free(shared_in);
        const shared_out = if (lw.shared_mlp) |smlp|
            try self.denseMLP(shared_in, &smlp)
        else
            try self.denseMLP(shared_in, &(switch (lw.mlp) {
                .dense => |dw| dw,
                .moe => unreachable,
            }));
        defer _ = mlx.mlx_array_free(shared_out);
        const shared_normed = try self.rmsNorm(shared_out, lw.post_ff_norm_1.?);
        defer _ = mlx.mlx_array_free(shared_normed);

        // Routed experts: router gets raw residual, experts get pre_ff_norm_2(residual)
        const expert_in = try self.rmsNorm(h, lw.pre_ff_norm_2.?);
        defer _ = mlx.mlx_array_free(expert_in);
        const expert_out = switch (lw.mlp) {
            .moe => |*mw| try self.moeMLP2(h, expert_in, mw),
            .dense => |*dw| try self.denseMLP(expert_in, dw),
        };
        defer _ = mlx.mlx_array_free(expert_out);
        const expert_normed = try self.rmsNorm(expert_out, lw.post_ff_norm_2.?);
        defer _ = mlx.mlx_array_free(expert_normed);

        if (std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null and trace_tail_once) {
            trace_tail_once = false;
            debugTraceHead("tail residual", h, self.s);
            debugTraceHead("tail shared_normed", shared_normed, self.s);
            debugTraceHead("tail expert_normed", expert_normed, self.s);
        }

        // Combine: shared + experts → post_ff_norm → residual add
        var combined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(combined);
        try mlx.check(mlx.mlx_add(&combined, shared_normed, expert_normed, self.s));
        const combined_normed = try self.rmsNorm(combined, lw.post_ff_norm.?);
        defer _ = mlx.mlx_array_free(combined_normed);
        var h_ff = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_ff, h, combined_normed, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_ff;

        // Layer scalar. DiffusionGemma's causal encoder pass uses its own
        // per-layer scalar (the only untied encoder text params); every
        // other caller uses the decoder/trunk scalar.
        const scalar = if (use_encoder_scalar and lw.encoder_layer_scalar != null)
            lw.encoder_layer_scalar
        else
            lw.layer_scalar;
        if (scalar) |ls| {
            var h_scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_scaled;
        }
        return h;
    }

    // ── DiffusionGemma bidirectional canvas decoder ──

    /// MLX_SERVE_DIFFUSION_TRACE=1 debugging aid: print the first 8 values
    /// of position 0 of a hidden state.
    var trace_tail_once: bool = true;
    fn debugTraceHead(label: []const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) void {
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        mlx.check(mlx.mlx_astype(&f, arr, .float32, s)) catch return;
        mlx.check(mlx.mlx_array_eval(f)) catch return;
        const d = mlx.mlx_array_data_float32(f) orelse return;
        std.debug.print("{s}: {any}\n", .{ label, d[0..8] });
    }

    /// Decoder attention over a noisy canvas: Q/K/V computed for the canvas
    /// only, RoPEd at absolute positions [offset, offset+L); the committed
    /// context K/V is READ from the cache (never written) and concatenated.
    /// Attention is fully bidirectional — no mask at all (B=1, no padding):
    /// full layers see the whole cache + canvas; sliding layers see the last
    /// `sliding_window − 1` cached positions + the whole canvas (enforced by
    /// slicing the cached K/V, mirroring mlx-vlm's O(window) trick — the HF
    /// reference zeroes the same region with a mask).
    fn diffusionDecoderAttn(
        self: *Transformer,
        ctx: *ForwardCtx,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const is_global = cfg.isGlobalLayer(layer);
        const h_count: c_int = @intCast(cfg.num_attention_heads);

        const cur_hd: u32 = cfg.layerHeadDim(layer);
        const cur_kv_h: u32 = cfg.layerKVHeads(layer);
        const q_shape = [_]c_int{ batch, seq_len, h_count, @intCast(cur_hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(cur_kv_h), @intCast(cur_hd) };
        const flat_shape = [_]c_int{ batch, seq_len, @intCast(@as(u32, @intCast(h_count)) * cur_hd) };

        const use_prop_rope = is_global and self.rope_freqs_global != null;
        const rope_dims: c_int = @intCast(cur_hd);
        const rope_base = mlx.mlx_optional_float{ .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq, .has_value = !use_prop_rope };
        const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
        const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };

        const perm = [_]c_int{ 0, 2, 1, 3 };
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Q projection + norm + RoPE
        const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_proj);
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
        const q_normed = try self.rmsNorm(q_r, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

        // K, V projections. On full layers fa.v_w aliases fa.k_w (no v_proj
        // in the checkpoint): V = param-free-norm(k_proj out) PRE-k_norm,
        // PRE-RoPE — same recompute the gemma4 MoE attention does.
        const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_proj);
        const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_proj);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        var v_after_norm = v_r;
        var v_normed_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_normed_arr);
        if (cfg.has_v_norm) {
            const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
            const vnw = if (has_dual_hd and is_global)
                (self.v_norm_weight_global orelse self.v_norm_weight.?)
            else
                self.v_norm_weight.?;
            v_normed_arr = try self.rmsNorm(v_r, vnw);
            v_after_norm = v_normed_arr;
        }

        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_after_norm, &perm, 4, self.s));

        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

        // Read the committed-context K/V from the cache WITHOUT writing.
        var full_k = k_rope;
        var full_v = v_t;
        var concat_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(concat_k);
        var concat_v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(concat_v);
        var kv_view: ?DenseKVView = null;
        defer if (kv_view) |*kv| kv.deinit();
        if (ctx.cache.seqLen(layer) > 0) {
            kv_view = try ctx.cache.denseView(layer, self.s);
            var cache_k = kv_view.?.k;
            var cache_v = kv_view.?.v;
            // Sliding layers: the canvas only attends to the last window−1
            // committed positions.
            var sliced_k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sliced_k);
            var sliced_v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sliced_v);
            if (!is_global and cfg.has_sliding_window) {
                const window: c_int = @as(c_int, @intCast(cfg.sliding_window)) - 1;
                const ck_shape = mlx.getShape(cache_k);
                const cache_len = ck_shape[2];
                if (window > 0 and cache_len > window) {
                    const start = [_]c_int{ 0, 0, cache_len - window, 0 };
                    const stop = [_]c_int{ ck_shape[0], ck_shape[1], cache_len, ck_shape[3] };
                    const strides = [_]c_int{ 1, 1, 1, 1 };
                    try mlx.check(mlx.mlx_slice(&sliced_k, cache_k, &start, 4, &stop, 4, &strides, 4, self.s));
                    try mlx.check(mlx.mlx_slice(&sliced_v, cache_v, &start, 4, &stop, 4, &strides, 4, self.s));
                    cache_k = sliced_k;
                    cache_v = sliced_v;
                }
            }
            const kvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(kvec);
            _ = mlx.mlx_vector_array_append_value(kvec, cache_k);
            _ = mlx.mlx_vector_array_append_value(kvec, k_rope);
            try mlx.check(mlx.mlx_concatenate_axis(&concat_k, kvec, 2, self.s));
            const vvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vvec);
            _ = mlx.mlx_vector_array_append_value(vvec, cache_v);
            _ = mlx.mlx_vector_array_append_value(vvec, v_t);
            try mlx.check(mlx.mlx_concatenate_axis(&concat_v, vvec, 2, self.s));
            full_k = concat_k;
            full_v = concat_v;
        }

        // Bidirectional SDPA — no mask (scale 1.0; q/k norms absorb it).
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, 1.0, "", none_mask, .{ .ctx = null }, self.s));

        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    /// DiffusionGemma decoder forward: one denoising step over the canvas.
    /// `canvas_ids` [1, L] are the current noisy token ids;
    /// `self_cond_embeddings` [1, L, H] is the previous step's soft embedding
    /// (probs @ embed_table × √H) or null on the first step. The KV cache is
    /// read-only here and `ctx.moe_seq_offset` does NOT advance — the canvas
    /// is re-forwarded every step until it commits, at which point the caller
    /// runs a causal ENCODER pass (forwardMoeWith with use_encoder_scalars)
    /// over the committed tokens to extend the cache.
    /// Returns softcapped logits [1, L, vocab]; caller frees.
    pub fn forwardDiffusionDecoder(
        self: *Transformer,
        ctx: *ForwardCtx,
        canvas_ids: mlx.mlx_array,
        self_cond_embeddings: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        const ml = self.moe_layers orelse return error.DiffusionRequiresMoeLayers;
        const sc = self.self_cond orelse return error.DiffusionWeightsMissing;
        const cfg = &self.config;
        const offset: c_int = @intCast(ctx.moe_seq_offset.*);

        // Canvas embeddings (× √hidden) + self-conditioning. Even with a
        // null signal the embeddings pass through the module's scale-free
        // post_norm (FFN(pre_norm(0)) ≡ 0), so layer-0 input is always
        // RMS-normalized.
        var h = try self.embedding(canvas_ids);
        if (self_cond_embeddings) |sig| {
            const dbg_sc = std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null;
            if (dbg_sc) debugTraceHead("sc sig", sig, self.s);
            const sig_normed = try self.rmsNorm(sig, sc.pre_norm);
            defer _ = mlx.mlx_array_free(sig_normed);
            if (dbg_sc) debugTraceHead("sc normed", sig_normed, self.s);
            const gate = try self.qmatmul(sig_normed, sc.gate_w, sc.gate_s, sc.gate_b);
            defer _ = mlx.mlx_array_free(gate);
            const up = try self.qmatmul(sig_normed, sc.up_w, sc.up_s, sc.up_b);
            defer _ = mlx.mlx_array_free(up);
            const act = try self.computeGeglu(gate, up);
            defer _ = mlx.mlx_array_free(act);
            const ffn_out = try self.qmatmul(act, sc.down_w, sc.down_s, sc.down_b);
            defer _ = mlx.mlx_array_free(ffn_out);
            if (dbg_sc) debugTraceHead("sc signal", ffn_out, self.s);
            var summed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(summed);
            try mlx.check(mlx.mlx_add(&summed, h, ffn_out, self.s));
            const post = try self.rmsNorm(summed, self.ones_hidden.?);
            _ = mlx.mlx_array_free(h);
            h = post;
            if (dbg_sc) debugTraceHead("sc out", h, self.s);
        } else {
            const post = try self.rmsNorm(h, self.ones_hidden.?);
            _ = mlx.mlx_array_free(h);
            h = post;
        }

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];

        const dbg = std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null;
        if (dbg) debugTraceHead("embed+sc", h, self.s);

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &ml[layer_idx];

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            const attn_out = switch (lw.attn) {
                .full => |fa| try self.diffusionDecoderAttn(ctx, normed, &fa, li, offset, batch, seq_len),
                .linear => return error.DiffusionUnsupportedLinearAttn,
            };
            defer _ = mlx.mlx_array_free(attn_out);
            if (dbg and layer_idx == 0) {
                debugTraceHead("layer0 normed", normed, self.s);
                debugTraceHead("layer0 attn_out", attn_out, self.s);
            }

            h = try self.gemma4MoeLayerTail(h, attn_out, lw, false);
            if (dbg and (layer_idx == 0 or layer_idx == 1 or layer_idx == 5 or layer_idx == 29)) {
                var buf: [32]u8 = undefined;
                const label = std.fmt.bufPrint(&buf, "after layer {d}", .{layer_idx}) catch "layer";
                debugTraceHead(label, h, self.s);
            }
        }

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);
        var logits = try self.lmHeadProject(final_normed);
        _ = mlx.mlx_array_free(final_normed);

        if (self.softcap_scalar != null) {
            const capped = try self.applySoftcap(logits);
            _ = mlx.mlx_array_free(logits);
            logits = capped;
        }
        return logits;
    }

    // ── Hybrid forward pass (LFM2, Nemotron-H) ──

    fn forwardHybridWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const hl = self.hybrid_layers.?;
        const offset = ctx.moe_seq_offset.*;
        const cfg = &self.config;

        var h = try self.embedding(token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];

        // Eval cadence: drop to per-layer when this chunk's score/dequant
        // transients are large — see prefillEvalCadence. LFM2/Nemotron-H ride
        // fused head dims, but a quantized KV cache's dense rebuild (and a
        // future head_dim-256 hybrid like qwen3_next) still counts.
        const hybrid_eval_cadence = prefillEvalCadence(
            MOE_EVAL_EVERY_N_LAYERS,
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            @intCast(seq_len),
            @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
            ctx.cache.config.scheme != .off,
        );

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &hl[layer_idx];

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Primary operation
            const op_out = switch (lw.op) {
                .gated_conv => |cw| try self.gatedConv(normed, &cw, &ctx.ssm_entries.?[layer_idx], batch, seq_len),
                .full_attn => |fa| try self.hybridAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, seq_len > 1),
                .mamba2 => |mw| try self.mamba2Mixer(normed, &mw, &ctx.ssm_entries.?[layer_idx], batch, seq_len),
                .dense_mlp => |dw| try self.denseMLP(normed, &dw),
                .simple_mlp => |sw| try self.simpleMLP(normed, &sw),
            };
            defer _ = mlx.mlx_array_free(op_out);

            // Residual connection
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, op_out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            // Optional MLP (LFM2: always present after mixer; Nemotron-H: null)
            if (lw.mlp) |mlp_w| {
                const ff_normed = try self.rmsNorm(h, lw.post_norm.?);
                defer _ = mlx.mlx_array_free(ff_normed);
                const mlp_out = try self.denseMLP(ff_normed, &mlp_w);
                defer _ = mlx.mlx_array_free(mlp_out);

                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            if (prefillEvalCadenceApplies(seq_len) and (layer_idx + 1) % hybrid_eval_cadence == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        ctx.moe_seq_offset.* += @intCast(seq_len);

        // Final norm (absent for LFM2)
        if (cfg.has_final_norm) {
            const final_normed = try self.rmsNorm(h, self.final_norm);
            _ = mlx.mlx_array_free(h);
            if (self.embedding_mode) return final_normed;
            const logits = try self.lmHeadProject(final_normed);
            _ = mlx.mlx_array_free(final_normed);
            return logits;
        } else {
            if (self.embedding_mode) return h;
            const logits = try self.lmHeadProject(h);
            _ = mlx.mlx_array_free(h);
            return logits;
        }
    }

    // ── Gated Convolution (LFM2) ──

    fn gatedConv(
        self: *Transformer,
        x: mlx.mlx_array,
        cw: *const GatedConvWeights,
        ssm: *SSMCacheEntry,
        batch: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        _ = seq_len;
        const hidden: c_int = @intCast(self.config.hidden_size);
        const kernel: c_int = @intCast(self.config.lfm_conv_kernel);

        // 1. Input projection: [B, S, hidden] → [B, S, 3*hidden]
        const proj = try self.qmatmul(x, cw.in_proj_w, cw.in_proj_s, cw.in_proj_b);
        defer _ = mlx.mlx_array_free(proj);

        // 2. Split into 3 equal parts: B, C, x (this order per mlx-lm/HF reference)
        const proj_shape = mlx.getShape(proj);
        const proj_seq = proj_shape[1];
        const strides3 = [_]c_int{ 1, 1, 1 };

        var b_gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b_gate);
        try mlx.check(mlx.mlx_slice(&b_gate, proj, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, proj_seq, hidden }, 3, &strides3, 3, self.s));

        var c_gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_gate);
        try mlx.check(mlx.mlx_slice(&c_gate, proj, &[_]c_int{ 0, 0, hidden }, 3, &[_]c_int{ batch, proj_seq, hidden * 2 }, 3, &strides3, 3, self.s));

        var x_conv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_conv);
        try mlx.check(mlx.mlx_slice(&x_conv, proj, &[_]c_int{ 0, 0, hidden * 2 }, 3, &[_]c_int{ batch, proj_seq, hidden * 3 }, 3, &strides3, 3, self.s));

        // 3. First gating: B * x
        var gated_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_input);
        try mlx.check(mlx.mlx_multiply(&gated_input, b_gate, x_conv, self.s));

        // 4. Conv1d with cache (depthwise, groups=hidden, no activation)
        const conv_out = try self.conv1dWithCache(gated_input, cw.conv_w, null, ssm, batch, hidden, kernel, false);
        defer _ = mlx.mlx_array_free(conv_out);

        // 5. Second gating: C_gate * conv_out
        var gated_output = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_output);
        try mlx.check(mlx.mlx_multiply(&gated_output, c_gate, conv_out, self.s));

        // 6. Output projection
        return self.qmatmul(gated_output, cw.out_proj_w, cw.out_proj_s, cw.out_proj_b);
    }

    // ── Mamba2 SSM (Nemotron-H) ──

    fn mamba2Mixer(
        self: *Transformer,
        x: mlx.mlx_array,
        mw: *const Mamba2Weights,
        ssm: *SSMCacheEntry,
        batch: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const num_heads: c_int = @intCast(cfg.mamba_num_heads);
        const head_dim: c_int = @intCast(cfg.mamba_head_dim);
        const n_groups: c_int = @intCast(cfg.mamba_n_groups);
        const state_size: c_int = @intCast(cfg.ssm_state_size);
        const d_inner: c_int = num_heads * head_dim; // intermediate_size
        const conv_dim: c_int = d_inner + 2 * n_groups * state_size;
        const kernel: c_int = @intCast(cfg.mamba_conv_kernel);
        const repeats: c_int = @divExact(num_heads, n_groups);

        // 1. Input projection: [B, S, hidden] → [B, S, d_inner + conv_dim + num_heads]
        const proj = try self.qmatmul(x, mw.in_proj_w, mw.in_proj_s, mw.in_proj_b);
        defer _ = mlx.mlx_array_free(proj);

        // 2. Split: gate [d_inner], conv_input [conv_dim], dt [num_heads]
        const strides3 = [_]c_int{ 1, 1, 1 };
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_slice(&gate, proj, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, d_inner }, 3, &strides3, 3, self.s));

        var conv_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_input);
        try mlx.check(mlx.mlx_slice(&conv_input, proj, &[_]c_int{ 0, 0, d_inner }, 3, &[_]c_int{ batch, seq_len, d_inner + conv_dim }, 3, &strides3, 3, self.s));

        var dt_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_raw);
        try mlx.check(mlx.mlx_slice(&dt_raw, proj, &[_]c_int{ 0, 0, d_inner + conv_dim }, 3, &[_]c_int{ batch, seq_len, d_inner + conv_dim + num_heads }, 3, &strides3, 3, self.s));

        // 3. Conv1d with cache + SiLU on conv_input
        const conv_out = try self.conv1dWithCache(conv_input, mw.conv1d_w, mw.conv1d_b, ssm, batch, conv_dim, kernel, true);
        defer _ = mlx.mlx_array_free(conv_out);

        // 4. Split conv output: x_ssm [d_inner], B [n_groups*state_size], C [n_groups*state_size]
        var x_ssm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_ssm);
        try mlx.check(mlx.mlx_slice(&x_ssm, conv_out, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, d_inner }, 3, &strides3, 3, self.s));

        const b_end: c_int = d_inner + n_groups * state_size;
        var B_proj = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_proj);
        try mlx.check(mlx.mlx_slice(&B_proj, conv_out, &[_]c_int{ 0, 0, d_inner }, 3, &[_]c_int{ batch, seq_len, b_end }, 3, &strides3, 3, self.s));

        var C_proj = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_proj);
        try mlx.check(mlx.mlx_slice(&C_proj, conv_out, &[_]c_int{ 0, 0, b_end }, 3, &[_]c_int{ batch, seq_len, conv_dim }, 3, &strides3, 3, self.s));

        // 5. Reshape to head format
        // x_ssm: [B, S, num_heads, head_dim]
        const x_shape = [_]c_int{ batch, seq_len, num_heads, head_dim };
        var x_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_h);
        try mlx.check(mlx.mlx_reshape(&x_h, x_ssm, &x_shape, 4, self.s));

        // B, C: [B, S, n_groups, state_size]
        const bc_shape = [_]c_int{ batch, seq_len, n_groups, state_size };
        var B_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(B_h);
        var C_h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(C_h);
        try mlx.check(mlx.mlx_reshape(&B_h, B_proj, &bc_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&C_h, C_proj, &bc_shape, 4, self.s));

        // 6. Compute dt = softplus(dt + dt_bias), clamp to time limits
        // Cast dt to float32 for precision (matching Python)
        var dt_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_f32);
        try mlx.check(mlx.mlx_astype(&dt_f32, dt_raw, .float32, self.s));
        // dt + dt_bias
        var dt_biased = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_biased);
        try mlx.check(mlx.mlx_add(&dt_biased, dt_f32, mw.dt_bias, self.s));
        // softplus: log1p(exp(x))
        var dt_exp_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_exp_val);
        try mlx.check(mlx.mlx_exp(&dt_exp_val, dt_biased, self.s));
        var dt_sp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_sp);
        try mlx.check(mlx.mlx_log1p(&dt_sp, dt_exp_val, self.s));
        // Clamp (use float32 scalars matching dt precision)
        var dt_min_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_min_arr);
        {
            const v = mlx.mlx_array_new_float(cfg.time_step_min);
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_astype(&dt_min_arr, v, .float32, self.s));
        }
        var dt_max_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_max_arr);
        {
            const v = mlx.mlx_array_new_float(cfg.time_step_max);
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_astype(&dt_max_arr, v, .float32, self.s));
        }
        var dt_clamped_lo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_clamped_lo);
        try mlx.check(mlx.mlx_maximum(&dt_clamped_lo, dt_sp, dt_min_arr, self.s));
        var dt_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_val);
        try mlx.check(mlx.mlx_minimum(&dt_val, dt_clamped_lo, dt_max_arr, self.s));

        // 7. A = -exp(A_log) — cast to float32 to match Python precision
        // Python's ssm_attn does: A = -mx.exp(A_log).astype(dt.dtype) where dt is float32.
        // Without this cast, A stays in BF16 and decay values dA = exp(A*dt) are imprecise,
        // compounding across 42 layers × N timesteps.
        var A_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_neg);
        {
            var A_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(A_exp);
            try mlx.check(mlx.mlx_exp(&A_exp, mw.A_log, self.s));
            var A_neg_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(A_neg_bf16);
            try mlx.check(mlx.mlx_negative(&A_neg_bf16, A_exp, self.s));
            try mlx.check(mlx.mlx_astype(&A_neg, A_neg_bf16, .float32, self.s));
        }

        // 8. Initialize SSM state if needed: [B, num_heads, head_dim, state_size]
        // Note: can't use ssm.initialized here — conv1dWithCache already set it to true.
        // Check if ssm_state is empty (ctx == null) as the actual init indicator.
        if (ssm.ssm_state.ctx == null) {
            const state_shape = [_]c_int{ batch, num_heads, head_dim, state_size };
            ssm.ssm_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, .float32, self.s));
        }

        // Precompute D reshaped for broadcasting: [H] → [H, 1]
        var D_bc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(D_bc);
        try mlx.check(mlx.mlx_expand_dims(&D_bc, mw.D, 1, self.s));

        // 9. Per-timestep SSM recurrence
        const T: usize = @intCast(seq_len);
        const out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);

        for (0..T) |t| {
            const ti: c_int = @intCast(t);

            // Extract timestep slices
            const strides4 = [_]c_int{ 1, 1, 1, 1 };
            // dt_t: [B, 1, num_heads] → [B, num_heads]
            var dt_t_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dt_t_3d);
            try mlx.check(mlx.mlx_slice(&dt_t_3d, dt_val, &[_]c_int{ 0, ti, 0 }, 3, &[_]c_int{ batch, ti + 1, num_heads }, 3, &strides3, 3, self.s));
            var dt_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dt_t);
            {
                const dt_reshape = [_]c_int{ batch, num_heads };
                try mlx.check(mlx.mlx_reshape(&dt_t, dt_t_3d, &dt_reshape, 2, self.s));
            }

            // x_t: [B, num_heads, head_dim]
            var x_t_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_t_4d);
            try mlx.check(mlx.mlx_slice(&x_t_4d, x_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, num_heads, head_dim }, 4, &strides4, 4, self.s));
            var x_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_t);
            {
                const x_reshape = [_]c_int{ batch, num_heads, head_dim };
                try mlx.check(mlx.mlx_reshape(&x_t, x_t_4d, &x_reshape, 3, self.s));
            }

            // B_t: [B, n_groups, state_size] → repeat to [B, num_heads, state_size]
            var B_t_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(B_t_4d);
            try mlx.check(mlx.mlx_slice(&B_t_4d, B_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, n_groups, state_size }, 4, &strides4, 4, self.s));
            var B_t_sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(B_t_sq);
            {
                const bc_rs = [_]c_int{ batch, n_groups, state_size };
                try mlx.check(mlx.mlx_reshape(&B_t_sq, B_t_4d, &bc_rs, 3, self.s));
            }
            var B_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(B_t);
            if (repeats > 1) {
                try mlx.check(mlx.mlx_repeat_axis(&B_t, B_t_sq, repeats, 1, self.s));
            } else {
                try mlx.check(mlx.mlx_array_set(&B_t, B_t_sq));
            }

            // C_t: same as B_t
            var C_t_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(C_t_4d);
            try mlx.check(mlx.mlx_slice(&C_t_4d, C_h, &[_]c_int{ 0, ti, 0, 0 }, 4, &[_]c_int{ batch, ti + 1, n_groups, state_size }, 4, &strides4, 4, self.s));
            var C_t_sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(C_t_sq);
            {
                const bc_rs = [_]c_int{ batch, n_groups, state_size };
                try mlx.check(mlx.mlx_reshape(&C_t_sq, C_t_4d, &bc_rs, 3, self.s));
            }
            var C_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(C_t);
            if (repeats > 1) {
                try mlx.check(mlx.mlx_repeat_axis(&C_t, C_t_sq, repeats, 1, self.s));
            } else {
                try mlx.check(mlx.mlx_array_set(&C_t, C_t_sq));
            }

            // dA = exp(A * dt_t): [B, num_heads]
            var A_dt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(A_dt);
            try mlx.check(mlx.mlx_multiply(&A_dt, A_neg, dt_t, self.s));
            var dA = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dA);
            try mlx.check(mlx.mlx_exp(&dA, A_dt, self.s));

            // dA_expanded: [B, num_heads, 1, 1] for state broadcast
            var dA_e1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dA_e1);
            var dA_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dA_exp);
            try mlx.check(mlx.mlx_expand_dims(&dA_e1, dA, 2, self.s));
            try mlx.check(mlx.mlx_expand_dims(&dA_exp, dA_e1, 3, self.s));

            // Decay state: state *= dA
            var decayed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(decayed);
            try mlx.check(mlx.mlx_multiply(&decayed, ssm.ssm_state, dA_exp, self.s));

            // dtx = x_t * dt_t: [B, num_heads, head_dim]
            var dt_exp2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dt_exp2);
            try mlx.check(mlx.mlx_expand_dims(&dt_exp2, dt_t, 2, self.s)); // [B, num_heads, 1]
            var dtx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dtx);
            try mlx.check(mlx.mlx_multiply(&dtx, x_t, dt_exp2, self.s));

            // Outer product update: dtx[..., :, None] * B_t[..., None, :]
            // dtx: [B, H, D] → [B, H, D, 1]
            // B_t: [B, H, S] → [B, H, 1, S]
            var dtx_e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dtx_e);
            try mlx.check(mlx.mlx_expand_dims(&dtx_e, dtx, 3, self.s));
            var B_t_e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(B_t_e);
            try mlx.check(mlx.mlx_expand_dims(&B_t_e, B_t, 2, self.s));
            var update = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(update);
            try mlx.check(mlx.mlx_multiply(&update, dtx_e, B_t_e, self.s));

            // new_state = decayed + update
            var new_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&new_state, decayed, update, self.s));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = new_state;

            // Output: y = sum_s(state * C_t) + x * D
            // C_t: [B, H, S] → [B, H, 1, S]
            var C_t_e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(C_t_e);
            try mlx.check(mlx.mlx_expand_dims(&C_t_e, C_t, 2, self.s));
            // state * C_t_e: [B, H, D, S]
            var state_c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state_c);
            try mlx.check(mlx.mlx_multiply(&state_c, ssm.ssm_state, C_t_e, self.s));
            // sum over S: [B, H, D]
            var y_state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(y_state);
            try mlx.check(mlx.mlx_sum_axis(&y_state, state_c, -1, false, self.s));

            // D * x: [B, H, D] where D_bc is [H, 1], x_t is [B, H, D]
            var dx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dx);
            try mlx.check(mlx.mlx_multiply(&dx, x_t, D_bc, self.s));

            // y_t = y_state + dx
            var y_t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&y_t, y_state, dx, self.s));
            _ = mlx.mlx_vector_array_append_value(out_vec, y_t);
            _ = mlx.mlx_array_free(y_t);

            if (t == 0) {
                try mlx.check(mlx.mlx_array_eval(ssm.ssm_state));
                log.debug("[mamba2] timestep 0 ok\n", .{});
            } else if ((t + 1) % RECURRENCE_EVAL_INTERVAL == 0) {
                try mlx.check(mlx.mlx_array_eval(ssm.ssm_state));
            }
        }

        // 10. Stack: [T, B, H, D] → transpose to [B, T, H, D]
        var stacked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stacked);
        try mlx.check(mlx.mlx_stack_axis(&stacked, out_vec, 0, self.s));
        const perm_tbhd = [_]c_int{ 1, 0, 2, 3 };
        var y_bthd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_bthd);
        try mlx.check(mlx.mlx_transpose_axes(&y_bthd, stacked, &perm_tbhd, 4, self.s));

        // Flatten heads: [B, T, H, D] → [B, T, H*D]
        const y_flat_shape = [_]c_int{ batch, seq_len, d_inner };
        var y_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_flat);
        try mlx.check(mlx.mlx_reshape(&y_flat, y_bthd, &y_flat_shape, 3, self.s));

        // 11. MambaRMSNormGated: silu(gate) * y, then group RMS norm, then weight
        // swiglu: silu(gate) * y
        const gated = try self.swiglu(gate, y_flat);
        defer _ = mlx.mlx_array_free(gated);

        // Group RMS norm: reshape to [B, S, n_groups, group_size], rms_norm per group, flatten
        // group_size = intermediate_size / n_groups (e.g. 7680/8 = 960), NOT head_dim
        const group_size: c_int = @divExact(d_inner, n_groups);
        const gated_shape = [_]c_int{ batch, seq_len, n_groups, group_size };
        var gated_grouped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_grouped);
        try mlx.check(mlx.mlx_reshape(&gated_grouped, gated, &gated_shape, 4, self.s));

        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        {
            // Parameter-free RMS norm: create ones weight of shape [group_size]
            const ones_shape = [_]c_int{group_size};
            var ones_w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones_w);
            try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, self.s));
            try mlx.check(mlx.mlx_fast_rms_norm(&normed, gated_grouped, ones_w, cfg.rms_norm_eps, self.s));
        }

        // Flatten back and apply weight
        var normed_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed_flat);
        try mlx.check(mlx.mlx_reshape(&normed_flat, normed, &y_flat_shape, 3, self.s));

        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, normed_flat, mw.norm_w, self.s));

        // 12. Output projection
        return self.qmatmul(weighted, mw.out_proj_w, mw.out_proj_s, mw.out_proj_b);
    }

    // ── Hybrid attention (LFM2, Nemotron-H) ──

    fn hybridAttnWith(
        self: *Transformer,
        ctx: *ForwardCtx,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer_idx: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
        is_prefill: bool,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const n_heads: c_int = @intCast(cfg.num_attention_heads);
        const n_kv_heads: c_int = @intCast(cfg.num_key_value_heads);
        const hd: c_int = @intCast(cfg.head_dim);
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

        // Q/K/V projections
        const q_raw = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_raw);
        const k_raw = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_raw);
        const v_raw = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_raw);

        // Reshape to heads: [B, S, n*hd] → [B, S, n, hd]
        const q_shape = [_]c_int{ batch, seq_len, n_heads, hd };
        const kv_shape = [_]c_int{ batch, seq_len, n_kv_heads, hd };
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_reshape(&q, q_raw, &q_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k, k_raw, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v, v_raw, &kv_shape, 4, self.s));

        // QK LayerNorm (LFM2 has it, Nemotron-H does not — checked by weight presence)
        if (cfg.has_qk_norm) {
            var q_normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rms_norm(&q_normed, q, fa.q_norm, cfg.rms_norm_eps, self.s));
            _ = mlx.mlx_array_free(q);
            q = q_normed;
            var k_normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rms_norm(&k_normed, k, fa.k_norm, cfg.rms_norm_eps, self.s));
            _ = mlx.mlx_array_free(k);
            k = k_normed;
        }

        // Transpose to [B, n, S, hd] for attention and RoPE
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        var v_t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v, &perm, 4, self.s));

        // RoPE (applied after transpose to [B, n, S, hd])
        const rope_base = mlx.mlx_optional_float.some(cfg.rope_theta);
        const no_freqs = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_rope(&q_t, q_t, hd, false, rope_base, cfg.rope_scaling_factor, offset, no_freqs, self.s));
        try mlx.check(mlx.mlx_fast_rope(&k_t, k_t, hd, false, rope_base, cfg.rope_scaling_factor, offset, no_freqs, self.s));

        // KV cache: update and get full K/V (DenseKVView owns its arrays only
        // in quant mode; in dense mode it aliases the cache view, so the defer
        // below is a no-op there).
        var kv_view = try ctx.cache.update(layer_idx, k_t, v_t, self.s, cfg.max_position_embeddings);
        defer kv_view.deinit();
        _ = mlx.mlx_array_free(k_t);
        _ = mlx.mlx_array_free(v_t);
        k_t = kv_view.k;
        v_t = kv_view.v;

        // Scaled dot-product attention (causal)
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (is_prefill) {
            if (try fusedSdpa256Prefill(self.s, q_t, k_t, v_t, attn_scale, 0)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            }
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
        }

        // Transpose back: [B, n, S, hd] → [B, S, n, hd] → [B, S, n*hd]
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
        const flat_shape = [_]c_int{ batch, seq_len, n_heads * hd };
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    // ── Simple (ungated) MLP with ReLU^2 (Nemotron-H) ──

    fn simpleMLP(self: *Transformer, x: mlx.mlx_array, sw: *const SimpleMlpWeights) !mlx.mlx_array {
        const up = try self.qmatmul(x, sw.up_w, sw.up_s, sw.up_b);
        defer _ = mlx.mlx_array_free(up);
        const activated = try self.reluSquared(up);
        defer _ = mlx.mlx_array_free(activated);
        return self.qmatmul(activated, sw.down_w, sw.down_s, sw.down_b);
    }

    /// Forward pass that returns hidden states (after final_norm, before lm_head).
    /// Output shape: [1, seq_len, hidden_size]. Caller must free.
    pub fn forwardEmbedding(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        return self.forwardEmbeddingMasked(token_ids, null);
    }

    /// Embedding forward for a padded [B, T] batch: `key_pad_mask` (additive,
    /// [B, 1, 1, T]) keeps padded positions out of encoder attention. Null
    /// mask == plain forwardEmbedding.
    pub fn forwardEmbeddingMasked(self: *Transformer, token_ids: mlx.mlx_array, key_pad_mask: ?mlx.mlx_array) !mlx.mlx_array {
        self.embedding_mode = true;
        defer self.embedding_mode = false;
        var ctx = self.defaultCtx();
        ctx.key_pad_mask = key_pad_mask;
        return self.forwardWith(&ctx, token_ids);
    }

    /// True when the checkpoint ships a sentence-transformers Dense head
    /// (EmbeddingGemma dense.0/dense.1), applied between pool and normalize.
    pub fn hasEmbedProjection(self: *const Transformer) bool {
        return self.dense0_w.ctx != null and self.dense1_w.ctx != null;
    }

    /// Sentence-transformers Dense head: pooled [B, H] → dense.0 → dense.1
    /// (identity activations, no layer bias — EmbeddingGemma's config).
    /// Input is cast to bf16 for the quantized matmuls. Caller frees.
    pub fn embedProjection(self: *const Transformer, pooled: mlx.mlx_array) !mlx.mlx_array {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, pooled, .bfloat16, self.s));
        const mid = try self.qmatmul(x, self.dense0_w, self.dense0_s, self.dense0_b);
        defer _ = mlx.mlx_array_free(mid);
        return self.qmatmul(mid, self.dense1_w, self.dense1_s, self.dense1_b);
    }

    /// Bidirectional batched encoder forward for embedding models built on a
    /// decoder arch (EmbeddingGemma: gemma3_text + use_bidirectional_attention).
    /// Self-contained, mirroring the BERT-arm precedent: no KV cache, no
    /// causality — full-attention layers attend over the whole (unpadded)
    /// sequence, sliding layers within a symmetric |i-j| < window band;
    /// `ctx.key_pad_mask` folds into both. Returns final-normed hidden
    /// [B, L, H] (this path only ever serves embeddings).
    fn forwardGemma3EncoderWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const cfg = &self.config;
        const h_count = cfg.num_attention_heads;
        const kv_h = cfg.num_key_value_heads;
        const hd = cfg.head_dim;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));

        var h = try self.embedding(token_ids);
        errdefer _ = mlx.mlx_array_free(h);
        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];

        const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
        const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };
        const perm = [_]c_int{ 0, 2, 1, 3 };

        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Additive masks, built once for the whole stack. Sequences within
        // the window need no band (equivalent to full attention).
        var band = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(band);
        if (cfg.has_sliding_window and seq_len > @as(c_int, @intCast(cfg.sliding_window))) {
            band = try encoderBandMask(self.allocator, @intCast(seq_len), cfg.sliding_window, self.s);
        }
        var band_pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(band_pad);
        if (band.ctx != null) {
            if (ctx.key_pad_mask) |kp| try mlx.check(mlx.mlx_add(&band_pad, band, kp, self.s));
        }
        const full_mask: mlx.mlx_array = ctx.key_pad_mask orelse none_mask;
        const sliding_mask: mlx.mlx_array = if (band_pad.ctx != null) band_pad else if (band.ctx != null) band else full_mask;

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &self.layers[layer_idx];
            // No cache exists here to share KV through; gemma3 never sets it.
            if (lw.kv_source != null) return error.UnsupportedEncoderArch;
            if (!cfg.has_pre_ff_norm) return error.UnsupportedEncoderArch;
            const is_global = cfg.isGlobalLayer(li);

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Q — projection, per-head norm, transpose, RoPE at 0..L-1.
            const q = try self.qmatmulMaybeBias(normed, lw.q_w, lw.q_s, lw.q_b, lw.q_bias);
            defer _ = mlx.mlx_array_free(q);
            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, &q_shape, 4, self.s));
            const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
            defer if (q_normed) |qn| {
                _ = mlx.mlx_array_free(qn);
            };
            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed orelse q_r, &perm, 4, self.s));

            // Same per-layer-type theta selection as the decoder forward.
            const use_prop_rope = is_global and self.rope_freqs_global != null;
            const rope_base_opt = mlx.mlx_optional_float{
                .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
                .has_value = !use_prop_rope,
            };
            const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
            const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };
            var q_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_rope);
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, @intCast(hd), false, rope_base_opt, rope_scale, 0, rope_freqs, self.s));

            // K, V
            const k = try self.qmatmulMaybeBias(normed, lw.k_w, lw.k_s, lw.k_b, lw.k_bias);
            defer _ = mlx.mlx_array_free(k);
            const v = if (lw.k_eq_v) k else try self.qmatmulMaybeBias(normed, lw.v_w, lw.v_s, lw.v_b, lw.v_bias);
            defer if (!lw.k_eq_v) {
                _ = mlx.mlx_array_free(v);
            };
            var k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_r);
            var v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_r);
            try mlx.check(mlx.mlx_reshape(&k_r, k, &kv_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&v_r, v, &kv_shape, 4, self.s));
            const k_normed: ?mlx.mlx_array = if (lw.k_norm) |kn| try self.rmsNorm(k_r, kn) else null;
            defer if (k_normed) |kn| {
                _ = mlx.mlx_array_free(kn);
            };
            var k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_t);
            var v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_t);
            try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed orelse k_r, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));
            var k_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_rope);
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, @intCast(hd), false, rope_base_opt, rope_scale, 0, rope_freqs, self.s));

            // Bidirectional SDPA (mlx composes internally for head_dim 256).
            const mask = if (is_global) full_mask else sliding_mask;
            var attn_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_out);
            if (mask.ctx != null) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, attn_scale, "array", mask, .{ .ctx = null }, self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_t, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            }

            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &out_shape, 3, self.s));
            const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
            defer _ = mlx.mlx_array_free(o_out);

            // Gemma sandwich norms + GeGLU MLP, exactly as the decoder.
            const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
            defer _ = mlx.mlx_array_free(attn_normed);
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
            defer _ = mlx.mlx_array_free(ff_normed);
            const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
            defer _ = mlx.mlx_array_free(gate_raw);
            const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
            defer _ = mlx.mlx_array_free(up);
            const gate_up = try self.computeGeglu(gate_raw, up);
            defer _ = mlx.mlx_array_free(gate_up);
            const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
            defer _ = mlx.mlx_array_free(down);
            const mlp_normed = try self.rmsNorm(down, lw.post_ff_norm.?);
            defer _ = mlx.mlx_array_free(mlp_normed);
            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;
        }

        const final = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);
        return final;
    }

    // ── Full Attention for MoE models (with optional output gate) ──

    /// Build per-token interleaved-M-RoPE cos/sin [1,1,seq_len,rope_dims] (bf16).
    /// Positions inside the prompt come from the explicit 3-D table; positions
    /// beyond it are generated text and collapse to scalar `absolute + delta`.
    /// `positions.base` lets a suffix-only speculative KV cache map its relative
    /// offsets back to the full prompt table.
    pub fn buildMropeCosSin(self: *Transformer, positions: mrope.PositionContext, offset: usize, seq_len: usize) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const cfg = &self.config;
        const rope_dims: usize = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const half = rope_dims / 2;

        const inv_freq = try self.allocator.alloc(f64, half);
        defer self.allocator.free(inv_freq);
        mrope.computeInvFreq(inv_freq, rope_dims, cfg.rope_theta);
        const sel = try self.allocator.alloc(u8, half);
        defer self.allocator.free(sel);
        mrope.interleavedSelector(sel, cfg.mrope_section);

        const cos_buf = try self.allocator.alloc(f32, seq_len * rope_dims);
        defer self.allocator.free(cos_buf);
        const sin_buf = try self.allocator.alloc(f32, seq_len * rope_dims);
        defer self.allocator.free(sin_buf);
        for (0..seq_len) |i| {
            const p = offset + i;
            const o = i * rope_dims;
            for (0..half) |d| {
                const axis: usize = sel[d];
                const pid: f64 = @floatFromInt(positions.axisPosition(axis, p));
                const angle = pid * inv_freq[d];
                const c: f32 = @floatCast(@cos(angle));
                const s: f32 = @floatCast(@sin(angle));
                // NeoX layout: cos/sin tiled across the two halves of rope_dims.
                cos_buf[o + d] = c;
                cos_buf[o + half + d] = c;
                sin_buf[o + d] = s;
                sin_buf[o + half + d] = s;
            }
        }
        const shape = [_]c_int{ 1, 1, @intCast(seq_len), @intCast(rope_dims) };
        const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(cf);
        const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(sf);
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cos, cf, .bfloat16, self.s));
        var sin = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sin, sf, .bfloat16, self.s));
        return .{ .cos = cos, .sin = sin };
    }

    /// Apply NeoX RoPE to the first `rope_dims` dims of `arr` [B,H,S,hd] using
    /// precomputed cos/sin [1,1,S,rope_dims] (broadcast over B,H); pass remaining
    /// dims through. Equivalent to `mlx_fast_rope(traditional=false)` but with
    /// per-token (M-RoPE) angles instead of a scalar offset.
    pub fn applyMrope(self: *Transformer, arr: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, rope_dims: c_int) !mlx.mlx_array {
        const sh = mlx.getShape(arr);
        const b = sh[0];
        const h = sh[1];
        const s = sh[2];
        const hd = sh[3];
        const half = @divExact(rope_dims, 2);
        const st = [_]c_int{ 1, 1, 1, 1 };

        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        try mlx.check(mlx.mlx_slice(&rot, arr, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));

        // rotate_half(rot) = concat(-rot[..,half:], rot[..,:half])
        var r2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_slice(&r2, rot, &[_]c_int{ 0, 0, 0, half }, 4, &[_]c_int{ b, h, s, rope_dims }, 4, &st, 4, self.s));
        var r1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_slice(&r1, rot, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, h, s, half }, 4, &st, 4, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, r2, self.s));
        const rh_arrs = [_]mlx.mlx_array{ neg, r1 };
        const rh_vec = mlx.mlx_vector_array_new_data(&rh_arrs, 2);
        defer _ = mlx.mlx_vector_array_free(rh_vec);
        var rh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rh);
        try mlx.check(mlx.mlx_concatenate_axis(&rh, rh_vec, -1, self.s));

        var xcos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xcos);
        try mlx.check(mlx.mlx_multiply(&xcos, rot, cos, self.s));
        var rsin = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rsin);
        try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
        var out_rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_rot);
        try mlx.check(mlx.mlx_add(&out_rot, xcos, rsin, self.s));

        if (hd == rope_dims) {
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, out_rot, .bfloat16, self.s));
            return out;
        }
        var pass = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pass);
        try mlx.check(mlx.mlx_slice(&pass, arr, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ b, h, s, hd }, 4, &st, 4, self.s));
        const cat_arrs = [_]mlx.mlx_array{ out_rot, pass };
        const cat_vec = mlx.mlx_vector_array_new_data(&cat_arrs, 2);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, cat_vec, -1, self.s));
        return out;
    }

    fn gatedFullAttnWith(
        self: *Transformer,
        ctx: *ForwardCtx,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
        is_prefill: bool,
    ) !mlx.mlx_array {
        const cache = ctx.cache;
        const cfg = &self.config;
        const h_count: c_int = @intCast(cfg.num_attention_heads);
        const kv_h: c_int = @intCast(cfg.num_key_value_heads);
        const hd: c_int = @intCast(cfg.head_dim);
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
        const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };

        // Q projection
        const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_proj);

        // With output gate: q_proj outputs [B, S, 2*H*D], split into queries + gate
        // Without: q_proj outputs [B, S, H*D], used directly as queries
        var queries: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(queries);
        var gate: mlx.mlx_array = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);

        if (cfg.attn_output_gate) {
            // Mirror mlx-lm qwen3_next.py:130-134: reshape Q-proj output to [B, S, H, D*2]
            // then `mx.split(_, 2, axis=-1)` into (queries, gate). The single split op
            // replaces our prior two-slice pattern (2 dispatches → 1 dispatch). Adds up
            // across all `full_attention_interval` layers — was the dominant Qwen 3.5/3.6
            // hybrid decode gap vs mlx-lm (5.7% → ~tied).
            const q_gate_shape = [_]c_int{ batch, seq_len, h_count, hd * 2 };
            var q_gate_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_gate_r);
            try mlx.check(mlx.mlx_reshape(&q_gate_r, q_proj, &q_gate_shape, 4, self.s));

            var split_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(split_vec);
            try mlx.check(mlx.mlx_split(&split_vec, q_gate_r, 2, -1, self.s));
            if (mlx.mlx_vector_array_size(split_vec) != 2) return error.UnexpectedSplitCount;

            queries = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&queries, split_vec, 0));

            var gate_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_4d);
            try mlx.check(mlx.mlx_vector_array_get(&gate_4d, split_vec, 1));

            try mlx.check(mlx.mlx_reshape(&gate, gate_4d, &flat_shape, 3, self.s));
        } else {
            const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
            queries = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&queries, q_proj, &q_shape, 4, self.s));
        }

        // K, V projections
        const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_proj);
        const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_proj);

        const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

        // Q/K norms
        const q_normed = try self.rmsNorm(queries, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // Transpose to [B, H, S, D]
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

        // Partial RoPE. Qwen3-VL image requests use interleaved M-RoPE: manual
        // per-token cos/sin on the prefill chunk (3D t/h/w angles at image
        // tokens), and scalar RoPE at offset+delta on decode (decode tokens are
        // text → t=h=w). Plain scalar partial RoPE otherwise (zero-cost for
        // text-only qwen3_5).
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        if (ctx.mrope_cos_cur) |cos| {
            const sin = ctx.mrope_sin_cur.?;
            q_rope = try self.applyMrope(q_t, cos, sin, rope_dims);
            k_rope = try self.applyMrope(k_t, cos, sin, rope_dims);
        } else {
            const eff_offset: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, eff_offset, .{ .ctx = null }, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, eff_offset, .{ .ctx = null }, self.s));
        }

        var kv_view = try cache.update(layer, k_rope, v_t, self.s, 0);
        defer kv_view.deinit();
        const full_k = kv_view.k;
        const full_v = kv_view.v;

        // Attention
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Fused-attn opt-in: see standard attention site for design notes.
        const sel_mode_moe: []const u8 = if (is_prefill) "causal" else "";
        if (ctx.kv_attn_fused and kv_view.has_quant_triple) {
            const fused = try kv_quant.quantAttention(
                q_rope,
                kv_view.kTriple(),
                kv_view.vTriple(),
                kv_view.bits,
                kv_view.group_size,
                attn_scale,
                sel_mode_moe,
                none_mask,
                self.s,
            );
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
        } else if (is_prefill) {
            if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            }
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
        }

        // Transpose back [B,H,S,D] -> [B,S,H*D]
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        // Optional output gating
        if (cfg.attn_output_gate) {
            var gate_sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_sig);
            try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, self.s));
            var gated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated);
            try mlx.check(mlx.mlx_multiply(&gated, attn_flat, gate_sig, self.s));
            return self.qmatmul(gated, fa.o_w, fa.o_s, fa.o_b);
        }

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    // ── Laguna attention (per-layer heads, YaRN/default RoPE, softplus gate) ──
    // Full-attention layers: 48 Q-heads, YaRN RoPE (theta 5e5, partial 0.5),
    // no sliding window. Sliding layers: 72 Q-heads, default RoPE (theta 1e4,
    // full rotary), 512-window mask. KV heads uniform (8). Per-head QK RMS-norm
    // before RoPE (qwen3 pattern); softplus per-head output gate before o_proj.
    // Mirrors gemma4MoeAttnWith's sliding-mask plumbing (head_dim 128 → no
    // fused hd-256 kernel). Reference: modeling_laguna.py LagunaAttention.
    fn lagunaAttnWith(
        self: *Transformer,
        ctx: *ForwardCtx,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
        is_prefill: bool,
        local_prefill_mask: *mlx.mlx_array,
        local_decode_mask: mlx.mlx_array,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        // "global" = full_attention layer (isGlobalLayer keys on layer_is_global).
        const is_full = cfg.isGlobalLayer(layer);
        const h_count: c_int = @intCast(cfg.layerNumHeads(layer)); // 48 full / 72 sliding
        const kv_h: c_int = @intCast(cfg.num_key_value_heads); // 8 uniform
        const hd: c_int = @intCast(cfg.head_dim); // 128
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
        const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
        const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
        const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const perm_back = [_]c_int{ 0, 2, 1, 3 };

        // Per-layer-type RoPE. Full layers: YaRN precomputed freqs (dims =
        // rotary_dim = head_dim*partial_global) + mscale on the rotated slice.
        // Sliding layers: default RoPE at rope_local_base_freq, full rotary.
        const use_yarn = is_full and self.rope_freqs_yarn != null;
        const rope_dims: c_int = if (use_yarn)
            @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor_global)
        else
            @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const rope_base = mlx.mlx_optional_float{
            .value = if (is_full) cfg.rope_theta else cfg.rope_local_base_freq,
            .has_value = !use_yarn, // yarn freqs override base
        };
        const rope_freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };

        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Q: proj → reshape → q_norm → transpose → rope
        const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_proj);
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
        const q_normed = try self.rmsNorm(q_r, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));

        // K: proj → reshape → k_norm → transpose → rope
        const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_proj);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, offset, rope_freqs, self.s));

        // YaRN mscale: scale the rotated slice of q/k by attention_factor. This
        // is exactly the reference's cos/sin *= attention_factor (the mscale
        // vector is 1.0 on the pass-through tail). Broadcast [head_dim] over BHS.
        if (use_yarn) {
            if (self.yarn_mscale) |ms| {
                var q_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&q_scaled, q_rope, ms, self.s));
                _ = mlx.mlx_array_free(q_rope);
                q_rope = q_scaled;
                var k_scaled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&k_scaled, k_rope, ms, self.s));
                _ = mlx.mlx_array_free(k_rope);
                k_rope = k_scaled;
            }
        }

        // V: proj → reshape → transpose
        const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_proj);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

        // KV cache: sliding layers trim to the window on DECODE (prefill keeps
        // the full buffer; the mask handles scope), full layers keep everything.
        const max_kv: u32 = if (is_full) 0 else cfg.sliding_window;
        var kv_view = try ctx.cache.update(layer, k_rope, v_t, self.s, max_kv);
        defer kv_view.deinit();
        const full_k = kv_view.k;
        const full_v = kv_view.v;

        // SDPA with sliding-window masking (mirrors gemma4MoeAttnWith exactly,
        // minus the fused hd-256 kernel — Laguna's head_dim is 128).
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (is_full) {
            if (is_prefill) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            }
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = offset + seq_len;
            if (is_prefill and total_kv <= sw) {
                // Window degenerates to plain causal — same kernel/reduction the
                // reference picks for short prompts.
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else if (is_prefill) {
                if (local_prefill_mask.ctx == null) {
                    local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, total_kv, sw);
                }
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask.*, .{ .ctx = null }, self.s));
            } else if (@as(c_int, @intCast(ctx.cache.seqLen(layer))) <= sw) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_decode_mask, .{ .ctx = null }, self.s));
            }
        }

        // [B,H,S,D] → [B,S,H*D]
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        // Softplus per-head output gate BEFORE o_proj (reference LagunaAttention):
        //   g = softplus(g_proj(x)) in fp32, cast to attn dtype (per-head scalar)
        //   attn = (attn.reshape[B,S,H,D] * g[...,None]).reshape[B,S,H*D]
        // softplus(x) = logaddexp(0, x) (numerically stable).
        if (cfg.laguna_attn_gate and fa.g_w.ctx != null) {
            const g_logits = try self.qmatmul(x, fa.g_w, fa.g_s, fa.g_b); // [B,S,h_count]
            defer _ = mlx.mlx_array_free(g_logits);
            var g_f32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_f32);
            try mlx.check(mlx.mlx_astype(&g_f32, g_logits, .float32, self.s));
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            var g_sp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_sp);
            try mlx.check(mlx.mlx_logaddexp(&g_sp, zero, g_f32, self.s));
            // Cast gate to attn dtype (bf16), then per-head broadcast multiply.
            var g_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_bf16);
            try mlx.check(mlx.mlx_astype(&g_bf16, g_sp, .bfloat16, self.s));
            const g_shape = [_]c_int{ batch, seq_len, h_count, 1 };
            var g_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_r);
            try mlx.check(mlx.mlx_reshape(&g_r, g_bf16, &g_shape, 4, self.s));
            var attn_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_4d);
            try mlx.check(mlx.mlx_reshape(&attn_4d, attn_flat, &q_shape, 4, self.s));
            var gated_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated_4d);
            try mlx.check(mlx.mlx_multiply(&gated_4d, attn_4d, g_r, self.s));
            var gated_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated_flat);
            try mlx.check(mlx.mlx_reshape(&gated_flat, gated_4d, &flat_shape, 3, self.s));
            return self.qmatmul(gated_flat, fa.o_w, fa.o_s, fa.o_b);
        }

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    // ── Gemma 4 Full Attention for MoE layers ──
    // Handles dual head dims, v_norm, sliding window, per-layer RoPE.

    fn gemma4MoeAttnWith(
        self: *Transformer,
        ctx: *ForwardCtx,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
        is_prefill: bool,
        // Pointer: the caller may have SKIPPED the eager sliding-mask build
        // (fused hd-256 kernel expected to band-mask in-kernel); if a per-call
        // check declines, the mask is built here once and cached back.
        local_prefill_mask: *mlx.mlx_array,
        local_decode_mask: mlx.mlx_array,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const is_global = cfg.isGlobalLayer(layer);
        const h_count: c_int = @intCast(cfg.num_attention_heads);

        // Per-layer dimensions
        const cur_hd: u32 = cfg.layerHeadDim(layer);
        const cur_kv_h: u32 = cfg.layerKVHeads(layer);
        const q_shape = [_]c_int{ batch, seq_len, h_count, @intCast(cur_hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(cur_kv_h), @intCast(cur_hd) };
        const flat_shape = [_]c_int{ batch, seq_len, @intCast(@as(u32, @intCast(h_count)) * cur_hd) };

        // Per-layer RoPE
        // RoPE: proportional for global layers (custom freqs), standard for local
        const use_prop_rope = is_global and self.rope_freqs_global != null;
        const rope_dims: c_int = @intCast(cur_hd); // full head dim for proportional (freqs handle partial)
        const rope_base = mlx.mlx_optional_float{ .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq, .has_value = !use_prop_rope };
        const rope_scale: f32 = if (use_prop_rope) 1.0 else if (is_global) (1.0 / cfg.rope_scaling_factor) else 1.0;
        const rope_freqs: mlx.mlx_array = if (use_prop_rope) self.rope_freqs_global.? else .{ .ctx = null };

        // Gemma 4: scale = 1.0 (QK-norm handles normalization)
        const attn_scale: f32 = 1.0;

        const perm = [_]c_int{ 0, 2, 1, 3 };
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Q projection + norm + RoPE
        const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_proj);
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, self.s));
        const q_normed = try self.rmsNorm(q_r, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

        // K, V projections
        const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_proj);
        const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_proj);
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

        // K norm
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // V norm (parameter-free RMS norm)
        var v_after_norm = v_r;
        var v_normed_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_normed_arr);
        if (cfg.has_v_norm) {
            const has_dual_hd = cfg.global_head_dim > 0 and cfg.global_head_dim != cfg.head_dim;
            const vnw = if (has_dual_hd and is_global)
                (self.v_norm_weight_global orelse self.v_norm_weight.?)
            else
                self.v_norm_weight.?;
            v_normed_arr = try self.rmsNorm(v_r, vnw);
            v_after_norm = v_normed_arr;
        }

        // Transpose K, V to [B, H, S, D]
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_after_norm, &perm, 4, self.s));

        // RoPE on K
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, rope_scale, offset, rope_freqs, self.s));

        // Update KV cache (trim to sliding window for local layers)
        const max_kv: u32 = if (is_global) 0 else if (cfg.has_sliding_window) cfg.sliding_window else 0;
        var kv_view = try ctx.cache.update(layer, k_rope, v_t, self.s, max_kv);
        defer kv_view.deinit();
        const full_k = kv_view.k;
        const full_v = kv_view.v;

        // Scaled dot-product attention with sliding window masking
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);

        // Causal prefill arms first try the fused hd-256 flash kernel (null =
        // precondition miss -> composed fallback, identical to before).
        if (!cfg.has_sliding_window) {
            if (is_prefill) {
                if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                    _ = mlx.mlx_array_free(attn_out);
                    attn_out = fused;
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                }
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            }
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = offset + seq_len;
            if (is_prefill and (is_global or total_kv <= sw)) {
                // Global layers, or everything fits in the window (the sliding
                // mask degenerates to plain causal — same kernel and reduction
                // order the mlx-lm/mlx-vlm reference picks for short prompts,
                // which keeps near-tie MoE router decisions from flipping vs
                // the reference; DiffusionGemma parity).
                if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
                    _ = mlx.mlx_array_free(attn_out);
                    attn_out = fused;
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                }
            } else if (is_prefill) {
                // Sliding-window prefill: band mask runs in-kernel when fused.
                if (try fusedSdpa256Prefill(self.s, q_rope, full_k, full_v, attn_scale, sw)) |fused| {
                    _ = mlx.mlx_array_free(attn_out);
                    attn_out = fused;
                } else {
                    if (local_prefill_mask.ctx == null) {
                        // Eager build was skipped for the fused path but this
                        // call declined — build once, cache for later layers.
                        local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, total_kv, sw);
                    }
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask.*, .{ .ctx = null }, self.s));
                }
            } else if (is_global) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            } else if (@as(c_int, @intCast(ctx.cache.seqLen(layer))) <= sw) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_decode_mask, .{ .ctx = null }, self.s));
            }
        }

        // Reshape: [B,H,S,D] → [B,S,H*D]
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    // ── GatedDeltaNet (linear attention layers) ──

    fn gatedDeltaNet(
        self: *Transformer,
        x: mlx.mlx_array,
        la: *const LinearAttnWeights,
        ssm: *SSMCacheEntry,
        batch: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const num_k_heads: c_int = @intCast(cfg.linear_num_key_heads);
        const num_v_heads: c_int = @intCast(cfg.linear_num_value_heads);
        const dk: c_int = @intCast(cfg.linear_key_head_dim);
        const dv: c_int = @intCast(cfg.linear_value_head_dim);
        const key_dim: c_int = dk * num_k_heads;
        const value_dim: c_int = dv * num_v_heads;
        const conv_dim: c_int = key_dim * 2 + value_dim;
        const kernel: c_int = @intCast(cfg.linear_conv_kernel_dim);

        // Projections: combined (qkvz+ba) or separate (qkv+z+a+b)
        var qkv: mlx.mlx_array = undefined;
        var z_proj: mlx.mlx_array = undefined;
        var a_proj: mlx.mlx_array = undefined;
        var b_proj: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(qkv);
        defer _ = mlx.mlx_array_free(z_proj);
        defer _ = mlx.mlx_array_free(a_proj);
        defer _ = mlx.mlx_array_free(b_proj);

        if (la.combined_proj) {
            // Combined QKVZ: output is interleaved by key-head groups.
            // Reshape to [B, S, nk, per_head], split per-head into q/k/v/z, then flatten back.
            const vph = @divExact(num_v_heads, num_k_heads); // value heads per key head group
            const qkvz_raw = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
            defer _ = mlx.mlx_array_free(qkvz_raw);
            const per_head = dk + dk + vph * dv + vph * dv;
            const gh_shape = [_]c_int{ batch, seq_len, num_k_heads, per_head };
            var qkvz_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qkvz_g);
            try mlx.check(mlx.mlx_reshape(&qkvz_g, qkvz_raw, &gh_shape, 4, self.s));

            const strides4 = [_]c_int{ 1, 1, 1, 1 };
            // q: [B,S,nk,dk]
            var q_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_g);
            try mlx.check(mlx.mlx_slice(&q_g, qkvz_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk }, 4, &strides4, 4, self.s));
            // k: [B,S,nk,dk]
            var k_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_g);
            try mlx.check(mlx.mlx_slice(&k_g, qkvz_g, &[_]c_int{ 0, 0, 0, dk }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk * 2 }, 4, &strides4, 4, self.s));
            // v: [B,S,nk,vph*dv]
            const v_off = dk * 2;
            const v_end = v_off + vph * dv;
            var v_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_g);
            try mlx.check(mlx.mlx_slice(&v_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_off }, 4, &[_]c_int{ batch, seq_len, num_k_heads, v_end }, 4, &strides4, 4, self.s));
            // z: [B,S,nk,vph*dv]
            var z_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(z_g);
            try mlx.check(mlx.mlx_slice(&z_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_end }, 4, &[_]c_int{ batch, seq_len, num_k_heads, per_head }, 4, &strides4, 4, self.s));

            // Flatten: q/k -> [B,S,key_dim], v/z -> [B,S,value_dim]
            const flat3_qk = [_]c_int{ batch, seq_len, key_dim };
            const flat3_vz = [_]c_int{ batch, seq_len, value_dim };
            var q_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_flat);
            var k_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_flat);
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            try mlx.check(mlx.mlx_reshape(&q_flat, q_g, &flat3_qk, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&k_flat, k_g, &flat3_qk, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&v_flat, v_g, &flat3_vz, 3, self.s));
            z_proj = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&z_proj, z_g, &flat3_vz, 3, self.s));

            // Concatenate [q, k, v] -> qkv [B,S,conv_dim]
            const qkv_arr = [_]mlx.mlx_array{ q_flat, k_flat, v_flat };
            const qkv_vec = mlx.mlx_vector_array_new_data(&qkv_arr, 3);
            defer _ = mlx.mlx_vector_array_free(qkv_vec);
            qkv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&qkv, qkv_vec, 2, self.s));

            // Combined BA: interleaved by key-head groups
            const ba_raw = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
            defer _ = mlx.mlx_array_free(ba_raw);
            const ba_per_head = vph * 2;
            const ba_shape = [_]c_int{ batch, seq_len, num_k_heads, ba_per_head };
            var ba_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ba_g);
            try mlx.check(mlx.mlx_reshape(&ba_g, ba_raw, &ba_shape, 4, self.s));
            var b_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(b_g);
            var a_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(a_g);
            try mlx.check(mlx.mlx_slice(&b_g, ba_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, vph }, 4, &strides4, 4, self.s));
            try mlx.check(mlx.mlx_slice(&a_g, ba_g, &[_]c_int{ 0, 0, 0, vph }, 4, &[_]c_int{ batch, seq_len, num_k_heads, ba_per_head }, 4, &strides4, 4, self.s));
            const flat3_ba = [_]c_int{ batch, seq_len, num_v_heads };
            b_proj = mlx.mlx_array_new();
            a_proj = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&b_proj, b_g, &flat3_ba, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&a_proj, a_g, &flat3_ba, 3, self.s));
        } else {
            qkv = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
            z_proj = try self.qmatmul(x, la.z_w, la.z_s, la.z_b);
            a_proj = try self.qmatmul(x, la.a_w, la.a_s, la.a_b);
            b_proj = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
        }
        // Conv1d with cache: prepend conv_state, apply depthwise conv + silu
        const conv_out = try self.conv1dWithCache(qkv, la.conv1d_w, null, ssm, batch, conv_dim, kernel, true);
        defer _ = mlx.mlx_array_free(conv_out);

        // Split conv output into Q, K, V
        // Q: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // K: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // V: [B, S, value_dim] → [B, S, num_v_heads, dv]
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ batch, seq_len, key_dim };
            try mlx.check(mlx.mlx_slice(&q_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 };
            try mlx.check(mlx.mlx_slice(&k_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }

        // Reshape to head dims
        const q_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const k_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const v_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var q_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_heads);
        var k_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_heads);
        var v_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_heads);
        try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &q_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k_heads, k_flat, &k_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, self.s));

        // Q/K normalization: q = (1/dk) * rms_norm(q, null), k = (1/sqrt(dk)) * rms_norm(k, null)
        // Scale scalars + the parameter-free rms_norm ones-weight (mlx-c
        // requires a non-empty weight) are cached on the Transformer — they
        // used to be rebuilt every layer every decode step.
        if (self.gdn_q_scale == null) {
            const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
            self.gdn_q_scale = bf16Scalar(inv_scale, self.s);
            self.gdn_k_scale = bf16Scalar(@sqrt(inv_scale), self.s);
        }
        const inv_scale_sq = self.gdn_q_scale.?;
        const inv_sqrt_sc = self.gdn_k_scale.?;
        if (self.gdn_ones_w == null) {
            const ones_shape = [_]c_int{dk};
            var w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&w, &ones_shape, 1, .bfloat16, self.s));
            self.gdn_ones_w = w;
        }
        const ones_w = self.gdn_ones_w.?;

        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, ones_w, 1e-6, self.s));
        var q_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_scaled);
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, inv_scale_sq, self.s));

        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, ones_w, 1e-6, self.s));
        var k_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_scaled);
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, inv_sqrt_sc, self.s));

        // Gating: g = exp(-exp(A_log) * softplus(a + dt_bias)) — one fused
        // kernel via compiled closure (mirrors mlx-lm's compute_g), raw chain
        // as fallback. [B, S, Hv]
        const g = try self.computeGdnGate(la.A_log, a_proj, la.dt_bias);
        defer _ = mlx.mlx_array_free(g);

        // beta = sigmoid(b)
        var beta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta);
        try mlx.check(mlx.mlx_sigmoid(&beta, b_proj, self.s)); // [B, S, Hv]

        // Initialize SSM state if needed.
        // Can't use ssm.initialized — conv1dWithCache already set it to true.
        // Check if ssm_state is empty (ctx == null) as the actual init indicator.
        if (ssm.ssm_state.ctx == null) {
            const state_shape = [_]c_int{ batch, num_v_heads, dv, dk };
            ssm.ssm_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, .bfloat16, self.s));
        }

        // Fused Metal kernel: runs the full T-step delta recurrence in one dispatch.
        // Inputs (shapes): q,k [B,T,Hk,Dk] (GQA handled in kernel), v [B,T,Hv,Dv],
        //                  g,beta [B,T,Hv], state_in [B,Hv,Dv,Dk], T scalar.
        // Outputs: y [B,T,Hv,Dv], state_out [B,Hv,Dv,Dk].
        const T_scalar = mlx.mlx_array_new_int(seq_len);
        defer _ = mlx.mlx_array_free(T_scalar);

        const y_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));

        var y_bthd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_bthd);

        if (self.spec_capture_ssm) {
            // Spec verify pass: emit per-position states so partial-accept
            // rollback needs no re-forward. state_seq: [T, B, Hv, Dv, Dk]
            // (last row unwritten — capture-tail trim), final state as its
            // own [B, Hv, Dv, Dk] output.
            const state_seq_shape = [_]c_int{ seq_len, batch, num_v_heads, dv, dk };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_seq_shape, 5, .bfloat16));
            const state_out_shape = [_]c_int{ batch, num_v_heads, dv, dk };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_out_shape, 4, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * num_v_heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

            // seq_stride = per-timestep element stride into state_seq.
            const seq_stride = mlx.mlx_array_new_int(batch * num_v_heads * dv * dk);
            defer _ = mlx.mlx_array_free(seq_stride);
            const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar, seq_stride };
            const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
            defer _ = mlx.mlx_vector_array_free(inputs_vec);

            const gdn_kernel = try getGdnKernelSeq();
            var outputs_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(outputs_vec);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
            if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
            try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

            // Stash the per-position sequence for rollback (takes ownership).
            var state_seq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&state_seq, outputs_vec, 1));
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = state_seq;

            // Continue normal flow with the kernel's own final state — no
            // slice view into the capture buffer (a view would pin the whole
            // [T,...] buffer across rounds and add 2 graph nodes per layer).
            var final_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&final_state, outputs_vec, 2));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = final_state;
        } else {
            const state_shape_out = [_]c_int{ batch, num_v_heads, dv, dk };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape_out, 4, .bfloat16));
            // Blocked-seq kernel at prefill widths on the 128-Dk/32-aligned-Dv
            // geometry (oMLX port, ~2x per GDN layer at 16K). Decode (T==1),
            // spec verify widths, and off-geometry shapes keep the stock
            // per-token kernel; the PLD capture path above is untouched.
            const use_blocked = gdnBlockedEnabled() and gdnBlockedEligible(seq_len, dk, dv, num_k_heads, num_v_heads);
            if (use_blocked) {
                // Grid: (256*(Dv/32), Hv, B) threads; threadgroup (256,1,1).
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256 * @divExact(dv, 32), num_v_heads, batch));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
            } else {
                // Grid: (32, Dv, B*Hv) threads; threadgroup: (32, 4, 1). Matches mlx-lm.
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * num_v_heads));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
            }
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

            const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar };
            const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
            defer _ = mlx.mlx_vector_array_free(inputs_vec);

            const gdn_kernel = if (use_blocked) try getGdnKernelBlocked(gdnBlockT()) else try getGdnKernel();
            var outputs_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(outputs_vec);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
            if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
            try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

            var new_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&new_state, outputs_vec, 1));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = new_state;
        }

        // Reshape z to [B, S, Hv, Dv]
        const z_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var z_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z_heads);
        try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &z_shape, 4, self.s));

        // RMSNormGated: swiglu(z, rms_norm(y, norm_weight))
        const y_normed = try self.rmsNorm(y_bthd, la.norm_w);
        defer _ = mlx.mlx_array_free(y_normed);
        const out_gated = try self.swiglu(z_heads, y_normed);
        defer _ = mlx.mlx_array_free(out_gated);

        // Flatten [B, S, Hv, Dv] → [B, S, value_dim]
        const out_flat_shape = [_]c_int{ batch, seq_len, value_dim };
        var out_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_flat);
        try mlx.check(mlx.mlx_reshape(&out_flat, out_gated, &out_flat_shape, 3, self.s));

        return self.qmatmul(out_flat, la.out_w, la.out_s, la.out_b);
    }

    // ── Dense MLP (SwiGLU: SiLU(gate(x)) * up(x) -> down) ──

    fn denseMLP(self: *Transformer, x: mlx.mlx_array, dw: *const DenseMlpWeights) !mlx.mlx_array {
        const gate = try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        defer _ = mlx.mlx_array_free(gate);
        const up = try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
        defer _ = mlx.mlx_array_free(up);
        const activated = try self.computeGeglu(gate, up);
        defer _ = mlx.mlx_array_free(activated);
        return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
    }

    // ── Sparse MoE MLP ──

    pub fn moeMLP(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        return self.moeMLP2(x, x, mw);
    }

    /// Decode-width expert projection via take + batched quantized_matmul,
    /// dodging our libmlx's serialized gather_qmm. `x_bc` is the [K,1,in] input
    /// (per-expert broadcast); `inds_flat` is the [K] selected-expert index
    /// vector. Returns [K,1,out]. Caller owns the result.
    fn batchedExpertMm(self: *Transformer, x_bc: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, inds_flat: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode) !mlx.mlx_array {
        var w_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_k);
        try mlx.check(mlx.mlx_take_axis(&w_k, w, inds_flat, 0, self.s));
        var sc_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc_k);
        try mlx.check(mlx.mlx_take_axis(&sc_k, sc, inds_flat, 0, self.s));
        var bi_k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi_k);
        var bi_arg = mlx.mlx_array{ .ctx = null };
        if (bi.ctx != null) {
            try mlx.check(mlx.mlx_take_axis(&bi_k, bi, inds_flat, 0, self.s));
            bi_arg = bi_k;
        }
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&out, x_bc, w_k, sc_k, bi_arg, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), self.s));
        return out; // [K, 1, out]
    }

    /// Decode-width MoE expert compute through the in-place gather-qmv kernel
    /// (`gatherQmv`), which indexes the expert bank directly instead of
    /// materializing the top-K experts. On success writes [B,S,K,hidden] into
    /// `out` and returns true; returns false when any projection is outside the
    /// kernel's supported set (non-affine bank, 3/5/6-bit width, odd geometry),
    /// leaving `out` untouched so the caller runs the batched take path.
    fn moeDecodeGatherQmv(
        self: *Transformer,
        out: *mlx.mlx_array,
        expert_x: mlx.mlx_array,
        inds: mlx.mlx_array,
        mw: *const MoeMlpWeights,
        gate_qp: anytype,
        up_qp: anytype,
        down_qp: anytype,
        D: c_int,
        K: c_int,
        B: c_int,
        S: c_int,
    ) !bool {
        // Routing indices arrive as int32/uint32 [.., K]; the kernel reads a
        // flat uint32 [K].
        var inds_u32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds_u32);
        {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const kshape = [_]c_int{K};
            try mlx.check(mlx.mlx_reshape(&flat, inds, &kshape, 1, self.s));
            try mlx.check(mlx.mlx_astype(&inds_u32, flat, .uint32, self.s));
        }
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        const dshape = [_]c_int{D};
        try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &dshape, 1, self.s));

        // gate / up share the single token's hidden state.
        const gate_2d = try gatherQmv(self.s, x_flat, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, inds_u32, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(gate_2d);
        const up_2d = try gatherQmv(self.s, x_flat, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, inds_u32, up_qp.bits, up_qp.group_size, up_qp.mode, false) orelse return false;
        defer _ = mlx.mlx_array_free(up_2d);

        const inter = mlx.getShape(gate_2d)[1];
        const k1i = [_]c_int{ K, 1, inter };
        var gate_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out);
        try mlx.check(mlx.mlx_reshape(&gate_out, gate_2d, &k1i, 3, self.s));
        var up_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out);
        try mlx.check(mlx.mlx_reshape(&up_out, up_2d, &k1i, 3, self.s));

        const expert_act = try self.computeGeglu(gate_out, up_out); // [K,1,inter]
        defer _ = mlx.mlx_array_free(expert_act);

        // down: each expert consumes ITS OWN activation → per-expert x layout.
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        const ki = [_]c_int{ K, inter };
        try mlx.check(mlx.mlx_reshape(&act_2d, expert_act, &ki, 2, self.s));
        const down_2d = try gatherQmv(self.s, act_2d, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_u32, down_qp.bits, down_qp.group_size, down_qp.mode, true) orelse return false;
        defer _ = mlx.mlx_array_free(down_2d);

        const hidden = mlx.getShape(down_2d)[1];
        const bskh_shape = [_]c_int{ B, S, K, hidden };
        try mlx.check(mlx.mlx_reshape(out, down_2d, &bskh_shape, 4, self.s));
        // Engagement is COUNTED, never inferred: a silent fallback to the
        // batched take path is output-identical and would look like a null win.
        if (!gqmv_engaged) {
            gqmv_engaged = true;
            log.info("[moe] gather-qmv kernel engaged: E_topk={d} inter={d} hidden={d} bits={d} gs={d}\n", .{ K, mlx.getShape(gate_2d)[1], hidden, gate_qp.bits, gate_qp.group_size });
        }
        return true;
    }

    /// MoE MLP with separate router and expert inputs.
    /// router_x: input for routing (raw hidden states).
    /// expert_x: input for expert computation (possibly normalized).
    fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const cfg = &self.config;
        // Decode MoE-internals profiler (MLX_SERVE_DECODE_PROFILE=1, S==1 only).
        const moe_prof = decodeProfileEnabled() and mlx.getShape(expert_x)[1] == 1;
        var mclk: ProfClock = if (moe_prof) ProfClock.init() else undefined;
        // Per-expert-weight params: mixed-precision MoE checkpoints vary bits
        // (and, with non-affine modes, group size + mode) per weight — resolve
        // each individually. gate/up consume the hidden dim; down consumes the
        // expert intermediate dim.
        const gate_qp = self.quantParamsHinted(mw.switch_gate_w, mw.switch_gate_s, lastDim(expert_x));
        const up_qp = self.quantParamsHinted(mw.switch_up_w, mw.switch_up_s, lastDim(expert_x));
        const down_qp = self.quantParamsHinted(mw.switch_down_w, mw.switch_down_s, if (cfg.moe_intermediate_size > 0) cfg.moe_intermediate_size else null);

        // Router: compute logits and top-K selection
        var router_logits: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(router_logits);

        if (mw.router_scale) |rs| {
            // Sigma-MoE: rms_norm(x, router_scale, eps) then project.
            // `router_scale` is pre-folded with hidden_size^-0.5 at model-load time
            // (see initMoeLayers) — no per-layer multiply needed.
            var normed_input = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(normed_input);
            try mlx.check(mlx.mlx_fast_rms_norm(&normed_input, router_x, rs, cfg.rms_norm_eps, self.s));

            const router_qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(normed_input));
            router_logits = try qmatmulBits(normed_input, mw.router_w, mw.router_s, mw.router_b, router_qp.bits, router_qp.group_size, router_qp.mode, self.s);
        } else {
            // Qwen3.5: direct projection
            const router_qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(router_x));
            router_logits = try qmatmulBits(router_x, mw.router_w, mw.router_s, mw.router_b, router_qp.bits, router_qp.group_size, router_qp.mode, self.s);
        }

        // Top-K + softmax/renormalize as a single fused kernel (when compiled).
        // Hy3 (expert_bias bound): sigmoid+bias selection instead of softmax.
        const routed = if (mw.expert_bias) |bias|
            try self.computeHy3Routing(router_logits, bias)
        else
            try self.computeMoeRouting(router_logits);
        const inds = routed.inds;
        defer _ = mlx.mlx_array_free(inds);
        var norm_scores = routed.norm_scores;
        defer _ = mlx.mlx_array_free(norm_scores);

        // Sigma-MoE: per-expert scale on selected indices (pes[inds]).
        // Stays outside the closure: depends on per-layer weights.
        if (mw.per_expert_scale) |pes| {
            var selected_scales = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(selected_scales);
            try mlx.check(mlx.mlx_take(
                &selected_scales,
                pes,
                inds,
                self.s,
            ));
            var scaled_scores = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled_scores, norm_scores, selected_scales, self.s));
            _ = mlx.mlx_array_free(norm_scores);
            norm_scores = scaled_scores;
        }

        // Expert computation. Two paths:
        //
        //   Decode (S=1): per-expert gather_qmm with `rhs_indices=inds` shape
        //   [B,S,K]. Output is [B,S,K,1,inter]; each token reads K expert
        //   blocks from random offsets, but at S=1 there are at most K unique
        //   experts so HBM scatter is bounded.
        //
        //   Multi-position (S>1): mlx-lm's `_gather_sort` flow. Flatten inds →
        //   argsort globally → `lhs_indices = order // K` selects which token
        //   row to feed each sorted slot; `rhs_indices = inds[order]` selects
        //   the expert (now sorted, so consecutive slots hit the same expert
        //   block → one HBM stream). After down_proj, an inverse permutation
        //   restores the original [B,S,K] layout. Critical for drafter verify
        //   on MoE: at block_size=4 + top_k=8 the old `total_inds >= 64`
        //   threshold left verify (32 inds) on the slow scatter path while the
        //   sorted path's argsort overhead is negligible at that size.
        if (moe_prof) {
            try mlx.check(mlx.mlx_array_eval(inds));
            try mlx.check(mlx.mlx_array_eval(norm_scores));
            decode_prof.moe_router_ns += mclk.lap();
        }

        const x_shape = mlx.getShape(expert_x);
        const B = x_shape[0];
        const S = x_shape[1];
        const D = x_shape[x_shape.len - 1];
        const inds_shape = mlx.getShape(inds);
        const K = inds_shape[inds_shape.len - 1];
        const total_inds: c_int = B * S * K;
        const do_sort = S > 1 or total_inds >= 64;
        const no_idx = mlx.mlx_array{ .ctx = null };

        var down_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_out);

        if (do_sort) {
            // ── Global-sort prefill path ──

            // Flatten inds → [N] where N = B*S*K
            const flat_shape = [_]c_int{total_inds};
            var flat_inds = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_inds);
            try mlx.check(mlx.mlx_reshape(&flat_inds, inds, &flat_shape, 1, self.s));

            // order = argsort(flat_inds), inv_order = argsort(order)
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, self.s));
            var inv_order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inv_order);
            try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, self.s));

            // sorted_inds = flat_inds[order], shape [N]
            var sorted_inds = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted_inds);
            try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, self.s));

            // lhs_idx = order // K, shape [N] — picks the source token row
            const k_arr = mlx.mlx_array_new_int(K);
            defer _ = mlx.mlx_array_free(k_arr);
            var lhs_idx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lhs_idx);
            try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, self.s));

            // x_flat: [B,S,D] → [B*S, D]
            const bs_d_shape = [_]c_int{ B * S, D };
            var x_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_flat);
            try mlx.check(mlx.mlx_reshape(&x_flat, expert_x, &bs_d_shape, 2, self.s));

            // x_rep: gather rows by lhs_idx → [N, D], then expand to [N, 1, D]
            // for gather_qmm (it expects an inner singleton dim before the
            // contracted feature dim).
            var x_gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_gathered);
            try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, self.s));
            const n1d_shape = [_]c_int{ total_inds, 1, D };
            var x_rep = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_rep);
            try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &n1d_shape, 3, self.s));

            // gate / up gather_qmm: x_rep [N,1,D], rhs_indices=sorted_inds [N],
            // output [N,1,intermediate]. squeeze inner 1 → [N, intermediate].
            var gate_out_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out_3d);
            try gatherExpertMm(&gate_out_3d, x_rep, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, no_idx, sorted_inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, true, self.s);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_3d, self.s));

            var up_out_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out_3d);
            try gatherExpertMm(&up_out_3d, x_rep, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, no_idx, sorted_inds, up_qp.bits, up_qp.group_size, up_qp.mode, true, self.s);
            var up_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out);
            try mlx.check(mlx.mlx_squeeze(&up_out, up_out_3d, self.s));

            const expert_act = try self.computeGeglu(gate_out, up_out);
            defer _ = mlx.mlx_array_free(expert_act);

            // down: expand inner singleton → [N,1,intermediate] → gather_qmm → [N,1,hidden]
            var act_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act_exp);
            try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
            var down_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_3d);
            try gatherExpertMm(&down_3d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, sorted_inds, down_qp.bits, down_qp.group_size, down_qp.mode, true, self.s);
            var down_squeezed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_squeezed);
            try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, self.s)); // [N, hidden]

            // Inverse permute → original order, then reshape back to [B,S,K,hidden].
            var down_unsorted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_unsorted);
            try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, self.s));
            const hidden = mlx.getShape(down_unsorted)[1];
            const bskh_shape = [_]c_int{ B, S, K, hidden };
            try mlx.check(mlx.mlx_reshape(&down_out, down_unsorted, &bskh_shape, 4, self.s));
        } else if (B * S == 1 and useBatchedExpertDecode(self) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null and
            try self.moeDecodeGatherQmv(&down_out, expert_x, inds, mw, gate_qp, up_qp, down_qp, D, K, B, S))
        {
            // ── Decode fast path: in-place gather-qmv kernel ──
            // Reads the expert bank directly with GPU-resident indices, so it
            // moves the ideal 9.8 MB/projection instead of the take path's 3x
            // (µbench at Laguna's 201 MB bank: 37 us vs batched 72 vs stock
            // gather_qmm 349). Everything is done inside the predicate; a null
            // return (non-affine bank, unsupported width) falls through to the
            // batched take path below with no behaviour change.
        } else if (B * S == 1 and useBatchedExpertDecode(self) and mw.switch_gate_s.ctx != null and mw.switch_up_s.ctx != null and mw.switch_down_s.ctx != null) {
            // ── Decode fast path: take experts + batched quantized_matmul ──
            // Dodges our libmlx's serialized decode gather_qmm. A per-arch
            // VALIDATED opt-in (laguna by default) — NOT default-on-for-all-MoE:
            // on small-expert MoEs (gemma4/qwen3.6) the take-materialization is a
            // net loss vs gather. See `useBatchedExpertDecode` / the policy test.
            // Quantized experts only (dense bf16 has null scales → gather path).
            var inds_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inds_flat);
            const kshape = [_]c_int{K};
            try mlx.check(mlx.mlx_reshape(&inds_flat, inds, &kshape, 1, self.s));

            // x_bc: expert_x [B,S,D] → [1,1,D] → broadcast [K,1,D]
            var x_1d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_1d);
            const x1_shape = [_]c_int{ 1, 1, D };
            try mlx.check(mlx.mlx_reshape(&x_1d, expert_x, &x1_shape, 3, self.s));
            var x_bc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_bc);
            const bc_shape = [_]c_int{ K, 1, D };
            try mlx.check(mlx.mlx_broadcast_to(&x_bc, x_1d, &bc_shape, 3, self.s));

            const gate_out = try self.batchedExpertMm(x_bc, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, inds_flat, gate_qp.bits, gate_qp.group_size, gate_qp.mode); // [K,1,inter]
            defer _ = mlx.mlx_array_free(gate_out);
            const up_out = try self.batchedExpertMm(x_bc, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, inds_flat, up_qp.bits, up_qp.group_size, up_qp.mode); // [K,1,inter]
            defer _ = mlx.mlx_array_free(up_out);

            const expert_act = try self.computeGeglu(gate_out, up_out); // [K,1,inter]
            defer _ = mlx.mlx_array_free(expert_act);

            const down = try self.batchedExpertMm(expert_act, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, inds_flat, down_qp.bits, down_qp.group_size, down_qp.mode); // [K,1,hidden]
            defer _ = mlx.mlx_array_free(down);
            const hidden = mlx.getShape(down)[mlx.getShape(down).len - 1];
            const bskh_shape = [_]c_int{ B, S, K, hidden };
            try mlx.check(mlx.mlx_reshape(&down_out, down, &bskh_shape, 4, self.s)); // [B,S,K,hidden]
        } else {
            // ── Decode / small-prefill path (gather_qmm) ──
            const exp_shape = [_]c_int{ B, S, 1, 1, D };
            var x_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x_exp);
            try mlx.check(mlx.mlx_reshape(&x_exp, expert_x, &exp_shape, 5, self.s));

            var gate_out_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out_5d);
            try gatherExpertMm(&gate_out_5d, x_exp, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, no_idx, inds, gate_qp.bits, gate_qp.group_size, gate_qp.mode, false, self.s);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_5d, self.s));

            var up_out_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out_5d);
            try gatherExpertMm(&up_out_5d, x_exp, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, no_idx, inds, up_qp.bits, up_qp.group_size, up_qp.mode, false, self.s);
            var up_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out);
            try mlx.check(mlx.mlx_squeeze(&up_out, up_out_5d, self.s));

            const expert_act = try self.computeGeglu(gate_out, up_out);
            defer _ = mlx.mlx_array_free(expert_act);

            var act_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act_exp);
            try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));
            var down_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down_5d);
            try gatherExpertMm(&down_5d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, inds, down_qp.bits, down_qp.group_size, down_qp.mode, false, self.s);
            try mlx.check(mlx.mlx_squeeze(&down_out, down_5d, self.s)); // [B, S, K, hidden]
        }

        if (moe_prof) {
            try mlx.check(mlx.mlx_array_eval(down_out));
            decode_prof.moe_experts_ns += mclk.lap();
        }

        // Weight by scores: down_out * norm_scores[..., None] → sum over K
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        try mlx.check(mlx.mlx_expand_dims(&scores_exp, norm_scores, -1, self.s)); // [B, S, K, 1]
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_out, scores_exp, self.s));
        var expert_sum = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&expert_sum, weighted, -2, false, self.s)); // [B, S, hidden]

        // Hy3: shared expert ALWAYS added, no gate (reference MoE.__call__:
        // `y = y + self.shared_mlp(x)`). shared_gate_w carries a real handle
        // only when the checkpoint shipped mlp.shared_mlp.* weights.
        if (mw.shared_ungated) {
            if (mw.shared_gate_w.ctx == null) return expert_sum;
            defer _ = mlx.mlx_array_free(expert_sum);
            const sh_gate = try self.qmatmul(expert_x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b);
            defer _ = mlx.mlx_array_free(sh_gate);
            const sh_up = try self.qmatmul(expert_x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b);
            defer _ = mlx.mlx_array_free(sh_up);
            const sh_act = try self.computeGeglu(sh_gate, sh_up);
            defer _ = mlx.mlx_array_free(sh_act);
            const sh_down = try self.qmatmul(sh_act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
            defer _ = mlx.mlx_array_free(sh_down);
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&result, expert_sum, sh_down, self.s));
            if (moe_prof) {
                try mlx.check(mlx.mlx_array_eval(result));
                decode_prof.moe_shared_ns += mclk.lap();
            }
            return result;
        }

        // Gemma 4: shared expert is handled separately in forwardMoe, just return expert_sum
        if (mw.shared_expert_gate_w == null) return expert_sum;
        defer _ = mlx.mlx_array_free(expert_sum);

        // Qwen3.5: shared expert + gated combination
        const sh_gate = try self.qmatmul(expert_x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b);
        defer _ = mlx.mlx_array_free(sh_gate);
        const sh_up = try self.qmatmul(expert_x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b);
        defer _ = mlx.mlx_array_free(sh_up);
        const sh_act = try self.computeGeglu(sh_gate, sh_up);
        defer _ = mlx.mlx_array_free(sh_act);
        const sh_down = try self.qmatmul(sh_act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
        defer _ = mlx.mlx_array_free(sh_down);

        const seg_w = mw.shared_expert_gate_w.?;
        const seg_qp = self.quantParamsHinted(seg_w, mw.shared_expert_gate_s.?, lastDim(expert_x));
        const sh_gate_logit = try qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, seg_qp.mode, self.s);
        defer _ = mlx.mlx_array_free(sh_gate_logit);
        var sh_gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sh_gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&sh_gate_sig, sh_gate_logit, self.s));
        var shared_gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shared_gated);
        try mlx.check(mlx.mlx_multiply(&shared_gated, sh_gate_sig, sh_down, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&result, expert_sum, shared_gated, self.s));
        return result;
    }

    // ── Mask helpers ──

    fn createCausalMask(self: *const Transformer, q_len: c_int, kv_len: c_int) !mlx.mlx_array {
        const offset_val = kv_len - q_len;
        const shape = [_]c_int{ q_len, kv_len };
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_full(&ones, &shape, 2, self.one, .bfloat16, self.s));
        var upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upper);
        try mlx.check(mlx.mlx_triu(&upper, ones, offset_val + 1, self.s));
        var bool_upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bool_upper);
        try mlx.check(mlx.mlx_astype(&bool_upper, upper, .bool_, self.s));
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&mask, bool_upper, neg_inf, zero, self.s));
        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var mask_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mask_4d, mask, &mask_shape, 4, self.s));
        _ = mlx.mlx_array_free(mask);
        return mask_4d;
    }

    fn createSlidingWindowDecodeMask(self: *const Transformer, kv_len: c_int, window: c_int) !mlx.mlx_array {
        var positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positions);
        try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, self.s));
        const window_start = mlx.mlx_array_new_int(kv_len - window);
        defer _ = mlx.mlx_array_free(window_start);
        var too_old = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(too_old);
        try mlx.check(mlx.mlx_less(&too_old, positions, window_start, self.s));
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var sw_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_mask);
        try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, self.s));
        const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
        var mask_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, self.s));
        return mask_4d;
    }

    fn createSlidingWindowMask(self: *const Transformer, q_len: c_int, kv_len: c_int, window: c_int) !mlx.mlx_array {
        const causal = try self.createCausalMask(q_len, kv_len);
        defer _ = mlx.mlx_array_free(causal);
        const offset_val = kv_len - q_len;
        var row_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_idx);
        try mlx.check(mlx.mlx_arange(&row_idx, @floatFromInt(offset_val), @floatFromInt(offset_val + q_len), 1, .int32, self.s));
        var col_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_idx);
        try mlx.check(mlx.mlx_arange(&col_idx, 0, @floatFromInt(kv_len), 1, .int32, self.s));
        const row_shape = [_]c_int{ q_len, 1 };
        const col_shape = [_]c_int{ 1, kv_len };
        var row_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_r);
        var col_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_r);
        try mlx.check(mlx.mlx_reshape(&row_r, row_idx, &row_shape, 2, self.s));
        try mlx.check(mlx.mlx_reshape(&col_r, col_idx, &col_shape, 2, self.s));
        var dist = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dist);
        try mlx.check(mlx.mlx_subtract(&dist, row_r, col_r, self.s));
        const window_arr = mlx.mlx_array_new_int(window);
        defer _ = mlx.mlx_array_free(window_arr);
        var too_far = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(too_far);
        try mlx.check(mlx.mlx_greater_equal(&too_far, dist, window_arr, self.s));
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        var sw_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_mask);
        try mlx.check(mlx.mlx_where(&sw_mask, too_far, neg_inf, zero, self.s));
        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var sw_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_4d);
        try mlx.check(mlx.mlx_reshape(&sw_4d, sw_mask, &mask_shape, 4, self.s));
        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&combined, causal, sw_4d, self.s));
        return combined;
    }
};

// ── Init helpers ──

fn initStandardLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { layers: []LayerWeights, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);
    // Dense bf16 weights are pre-transposed at load; track the new arrays so
    // Transformer.deinit frees them. Empty (no allocations) for quantized models.
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

        // Gemma 4 KV-layer sharing: layers in the shared tail reuse an earlier
        // layer's K/V (the forward reads `kv_source`'s cache), so they carry no
        // k_proj/k_norm/v_proj of their own. Some exports physically drop those
        // tensors — load them only for non-shared layers.
        lw.kv_source = config.getKVSourceLayer(li);
        const kv_shared = lw.kv_source != null;

        const input_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.input_norm = if (config.norm_has_offset) try addOne(input_norm_raw, s) else input_norm_raw;
        const post_attn_raw = getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.post_attn_norm = if (config.norm_has_offset) try addOne(post_attn_raw, s) else post_attn_raw;

        if (config.has_pre_ff_norm) {
            const pre_ff_raw = getLayerWeight(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.pre_ff_norm = if (config.norm_has_offset) try addOne(pre_ff_raw, s) else pre_ff_raw;
            const post_ff_raw = getLayerWeight(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.post_ff_norm = if (config.norm_has_offset) try addOne(post_ff_raw, s) else post_ff_raw;
        } else {
            lw.pre_ff_norm = null;
            lw.post_ff_norm = null;
        }

        if (config.has_qk_norm) {
            const q_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight");
            lw.q_norm = if (config.norm_has_offset) try addOne(q_norm_raw, s) else q_norm_raw;
            // KV-shared layers compute no K, so they carry no k_norm.
            if (kv_shared) {
                lw.k_norm = null;
            } else {
                const k_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight");
                lw.k_norm = if (config.norm_has_offset) try addOne(k_norm_raw, s) else k_norm_raw;
            }
        } else {
            lw.q_norm = null;
            lw.k_norm = null;
        }

        lw.q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight");
        lw.q_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits);
        lw.q_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config);
        // Additive q-proj bias (Qwen2). Optional — empty for archs without it.
        lw.q_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.bias") orelse mlx.mlx_array_new();
        if (kv_shared) {
            // No own K/V — the forward reads kv_source's cache. Leave empty.
            lw.k_eq_v = false;
            lw.k_w = mlx.mlx_array_new();
            lw.k_s = mlx.mlx_array_new();
            lw.k_b = mlx.mlx_array_new();
            lw.k_bias = mlx.mlx_array_new();
            lw.v_w = mlx.mlx_array_new();
            lw.v_s = mlx.mlx_array_new();
            lw.v_b = mlx.mlx_array_new();
            lw.v_bias = mlx.mlx_array_new();
        } else {
            lw.k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            lw.k_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            lw.k_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            lw.k_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.bias") orelse mlx.mlx_array_new();
            // Gemma 4 (31B): full_attention layers share V with K (no v_proj weight stored).
            // Sliding_attention layers still have separate V.
            lw.k_eq_v = config.attention_k_eq_v and config.isGlobalLayer(li);
            if (lw.k_eq_v) {
                lw.v_w = lw.k_w;
                lw.v_s = lw.k_s;
                lw.v_b = lw.k_b;
                lw.v_bias = lw.k_bias;
            } else {
                lw.v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
                lw.v_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.v_proj.scales", config.quant_bits);
                lw.v_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", &config);
                lw.v_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.bias") orelse mlx.mlx_array_new();
            }
        }
        lw.o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight");
        lw.o_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits);
        lw.o_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", &config);

        lw.gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight");
        lw.gate_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.gate_proj.scales", config.quant_bits);
        lw.gate_b = getLayerBias(weights, name_buf, prefix, li, "mlp.gate_proj.biases", &config);
        lw.up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight");
        lw.up_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.up_proj.scales", config.quant_bits);
        lw.up_b = getLayerBias(weights, name_buf, prefix, li, "mlp.up_proj.biases", &config);
        lw.down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight");
        lw.down_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "mlp.down_proj.scales", config.quant_bits);
        lw.down_b = getLayerBias(weights, name_buf, prefix, li, "mlp.down_proj.biases", &config);

        // Dense bf16: pre-transpose [out,in]→[in,out] so qmatmulBits dispatches to
        // a plain matmul. No-ops on quantized weights (scales non-null).
        try maybeTransposeForBf16(&lw.q_w, lw.q_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.k_w, lw.k_s, &owned_bf16, allocator, s);
        if (lw.k_eq_v) {
            lw.v_w = lw.k_w; // re-alias V to the transposed K (no second copy)
        } else {
            try maybeTransposeForBf16(&lw.v_w, lw.v_s, &owned_bf16, allocator, s);
        }
        try maybeTransposeForBf16(&lw.o_w, lw.o_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.gate_w, lw.gate_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.up_w, lw.up_s, &owned_bf16, allocator, s);
        try maybeTransposeForBf16(&lw.down_w, lw.down_s, &owned_bf16, allocator, s);

        // Gemma 4: per-layer scalar
        lw.layer_scalar = getLayerWeightOpt(weights, name_buf, prefix, li, "layer_scalar");

        // Gemma 4: PLE per-layer weights. Must initialize even in the no-PLE case so the
        // optional tags are not read as uninitialized memory later in the eval loop
        // (the layers slice comes from `allocator.alloc` which skips struct defaults).
        lw.ple_gate_w = null;
        lw.ple_gate_s = null;
        lw.ple_gate_b = null;
        lw.ple_proj_w = null;
        lw.ple_proj_s = null;
        lw.ple_proj_b = null;
        lw.ple_norm = null;
        if (config.hidden_size_per_layer_input > 0) {
            lw.ple_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_input_gate.weight");
            // Dense bf16: scales/biases don't exist. The forward unwraps these with
            // `.?` then feeds qmatmul, so supply a null-ctx array (not Zig-null) →
            // qmatmulBits sees the bf16 path.
            lw.ple_gate_s = getLayerScaleOrEmptyOpt(weights, name_buf, prefix, li, "per_layer_input_gate.scales", config.quant_bits);
            lw.ple_gate_b = getLayerBias(weights, name_buf, prefix, li, "per_layer_input_gate.biases", &config);
            lw.ple_proj_w = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_projection.weight");
            lw.ple_proj_s = getLayerScaleOrEmptyOpt(weights, name_buf, prefix, li, "per_layer_projection.scales", config.quant_bits);
            lw.ple_proj_b = getLayerBias(weights, name_buf, prefix, li, "per_layer_projection.biases", &config);
            lw.ple_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_per_layer_input_norm.weight");
            // bf16: pre-transpose the two PLE projections (used via qmatmul).
            if (lw.ple_gate_w) |*w| try maybeTransposeForBf16(w, lw.ple_gate_s.?, &owned_bf16, allocator, s);
            if (lw.ple_proj_w) |*w| try maybeTransposeForBf16(w, lw.ple_proj_s.?, &owned_bf16, allocator, s);
        }
    }
    return .{ .layers = layers, .owned_bf16 = try owned_bf16.toOwnedSlice(allocator) };
}

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // Swap the last two axes for any rank: 2D weights [out, in] → [in, out];
    // stacked MoE expert tensors [experts, out, in] → [experts, in, out].
    const ndim = mlx.getShape(w).len;
    var perm: [8]c_int = undefined; // mlx arrays never exceed 8 dims here
    for (0..ndim) |i| perm[i] = @intCast(i);
    perm[ndim - 2] = @intCast(ndim - 1);
    perm[ndim - 1] = @intCast(ndim - 2);
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

/// In-place: if `*sc` is null-ctx, we treat the matching `*w` as plain bf16.
/// Replace `*w` with its pre-transposed `[in, out]` form and record the new
/// array in `owned` so we can free it on Transformer.deinit.
fn maybeTransposeForBf16(
    w: *mlx.mlx_array,
    sc: mlx.mlx_array,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !void {
    if (sc.ctx != null) return; // quantized weight — leave as-is
    if (w.ctx == null) return; // absent weight (e.g. KV-shared layer) — nothing to transpose
    const transposed = try transposeBf16Weight(w.*, s);
    w.* = transposed;
    try owned.append(allocator, transposed);
}

fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry, owned_bf16: []mlx.mlx_array } {
    log.info("Precomputing MoE layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const moe_layers = try allocator.alloc(MoeLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |a| _ = mlx.mlx_array_free(a);
        owned_bf16.deinit(allocator);
    }
    // DiffusionGemma reuses the Gemma 4 26B-A4B layer structure verbatim —
    // both take the gemma4 binding/forward arms here.
    const is_gemma4 = config.isGemma4Layers();
    const is_hy3 = std.mem.eql(u8, config.model_type, "hy_v3");
    const is_laguna = std.mem.eql(u8, config.model_type, "laguna");

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        const is_linear = config.isLinearLayer(li);

        lw.input_norm = getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.post_attn_norm = getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.is_linear = is_linear;

        // `moe_layers` comes from `allocator.alloc` which skips struct defaults, so every
        // optional must be initialized before the conditional Gemma-4-only assignments;
        // otherwise the eval loop reads uninitialized memory as valid handles (segfaults
        // with 0xaa...aa on Qwen3-Next and similar non-Gemma MoE models).
        lw.pre_ff_norm = null;
        lw.post_ff_norm = null;
        lw.pre_ff_norm_2 = null;
        lw.post_ff_norm_1 = null;
        lw.post_ff_norm_2 = null;
        lw.layer_scalar = null;
        lw.encoder_layer_scalar = null;
        lw.shared_mlp = null;

        // DiffusionGemma: the encoder's per-layer scalars are the only
        // untied encoder text params; absolute name, outside weight_prefix.
        if (config.isDiffusion()) {
            var enc_buf: [256]u8 = undefined;
            const enc_name = std.fmt.bufPrint(&enc_buf, "model.encoder.language_model.layers.{d}.layer_scalar", .{li}) catch unreachable;
            lw.encoder_layer_scalar = weights.get(enc_name);
        }

        // Gemma 4 MoE: extra feedforward norms, layer scalar, shared expert MLP
        if (is_gemma4) {
            lw.pre_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.post_ff_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.pre_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "pre_feedforward_layernorm_2.weight");
            lw.post_ff_norm_1 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_1.weight");
            lw.post_ff_norm_2 = getLayerWeightOpt(weights, name_buf, prefix, li, "post_feedforward_layernorm_2.weight");
            lw.layer_scalar = getLayerWeightOpt(weights, name_buf, prefix, li, "layer_scalar");
            lw.shared_mlp = .{
                .gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            };
            const sm = &lw.shared_mlp.?;
            try maybeTransposeForBf16(&sm.gate_w, sm.gate_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.up_w, sm.up_s, &owned_bf16, allocator, s);
            try maybeTransposeForBf16(&sm.down_w, sm.down_s, &owned_bf16, allocator, s);
        }

        if (is_linear) {
            // Detect combined (qkvz+ba) vs separate (qkv+z+a+b) projections.
            // Each projection's `*_s`/`*_b` are loaded optionally — Unsloth Dynamic
            // checkpoints (e.g. Qwen3.6 UD) leave linear_attn projections as plain
            // bf16 with no scales/biases tensors, even though the rest of the model
            // is quantized. Null-ctx scales triggers a transpose-on-load so that
            // `qmatmulBits` can dispatch to plain `mlx_matmul`.
            const combined = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight") != null;
            if (combined) {
                lw.attn = .{ .linear = .{
                    .combined_proj = true,
                    .qkv_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.biases") orelse mlx.mlx_array_new(),
                    .z_w = mlx.mlx_array_new(),
                    .z_s = mlx.mlx_array_new(),
                    .z_b = mlx.mlx_array_new(),
                    .a_w = mlx.mlx_array_new(),
                    .a_s = mlx.mlx_array_new(),
                    .a_b = mlx.mlx_array_new(),
                    .b_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
            } else {
                lw.attn = .{ .linear = .{
                    .qkv_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                    .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales") orelse mlx.mlx_array_new(),
                    .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases") orelse mlx.mlx_array_new(),
                    .z_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                    .z_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales") orelse mlx.mlx_array_new(),
                    .z_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases") orelse mlx.mlx_array_new(),
                    .a_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                    .a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales") orelse mlx.mlx_array_new(),
                    .a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases") orelse mlx.mlx_array_new(),
                    .b_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                    .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales") orelse mlx.mlx_array_new(),
                    .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                    .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
                const la = &lw.attn.linear;
                try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, allocator, s);
            }
        } else {
            // Laguna ships in two quant layouts: poolside nvfp4 (attention DENSE
            // bf16 — no scales) and mlx-lm affine (attention QUANTIZED — scales
            // present, q/k/v/o all U32+scales+biases). Probe per-tensor: scales
            // present → the quant path (quantParamsHinted resolves bits from
            // geometry); absent → bf16 (maybeTransposeForBf16). Other archs keep
            // mandating scales (config.quant_bits) so a genuinely-missing scale
            // still errors loudly. getLayerBias already tolerates absence.
            const k_s = if (is_laguna)
                (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array_new())
            else
                getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            const k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            // Gemma 4 MoE: global layers use K=V (no separate v_proj)
            const v_w = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.weight") orelse k_w;
            const v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse k_s;
            const v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse k_b;
            const v_aliases_k = v_w.ctx == k_w.ctx;
            lw.attn = .{ .full = .{
                .q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                .q_s = if (is_laguna)
                    (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array_new())
                else
                    getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits),
                .q_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                .k_w = k_w,
                .k_s = k_s,
                .k_b = k_b,
                .v_w = v_w,
                .v_s = v_s,
                .v_b = v_b,
                .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = if (is_laguna)
                    (getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array_new())
                else
                    getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits),
                .o_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", &config),
                .q_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight"),
                .k_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight"),
            } };
            {
                // Dense bf16 (null-ctx scales): pre-transpose [out,in]→[in,out] so
                // qmatmulBits dispatches to a plain matmul. No-op on quantized weights.
                const fa = &lw.attn.full;
                try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, allocator, s);
                if (v_aliases_k) {
                    // K=V share one weight — re-alias V to the transposed K (don't
                    // create a second copy or double-free).
                    fa.v_w = fa.k_w;
                } else {
                    try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, allocator, s);
                }
                // Laguna softplus per-head output gate (self_attn.g_proj →
                // per-head logits [B,S,n_heads]). bf16 in the nvfp4 layout
                // (scales absent → pre-transposed), quantized in the mlx-lm
                // affine layout (scales+biases present → quant path). Probe.
                if (is_laguna) {
                    fa.g_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.g_proj.weight");
                    fa.g_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.g_proj.scales") orelse mlx.mlx_array_new();
                    fa.g_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.g_proj.biases", &config);
                    try maybeTransposeForBf16(&fa.g_w, fa.g_s, &owned_bf16, allocator, s);
                }
            }
        }

        // hy_v3: layers [0, first_k_dense_replace) are DENSE (layer 0 on the
        // 295B) — they take the dense binding arm below. Every established
        // arch has first_k_dense_replace == 0, so layer_is_moe == isMoe() there.
        // Laguna: mlp_only_layers (layer 0) are dense; resolve by a per-layer
        // weight-presence probe (converter-proof — a future variant with a
        // non-prefix dense-layer set stays correct without a config field).
        const layer_is_moe = if (is_laguna)
            (getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.weight") != null)
        else
            (config.isMoe() and li >= config.first_k_dense_replace);
        if (layer_is_moe and is_gemma4) {
            // Gemma 4 MoE: different weight naming, Sigma-MoE routing, no shared expert gate.
            // Each `*_s`/`*_b` is loaded optionally for Unsloth Dynamic compatibility —
            // bf16 layers carry only the weight, no scales/biases. The post-construction
            // `maybeTransposeForBf16` calls are no-ops for already-quantized weights.
            //
            // Expert naming variants:
            //   Gemma 4 26B-A4B:  experts.switch_glu.{gate,up,down}_proj.*
            //   DiffusionGemma:   experts.gate_up_proj.* (PACKED [E, 2M, X],
            //                     gate rows first) + experts.down_proj.*
            // The packed variant is sliced into gate/up halves at load
            // (splitPackedGateUp) so both feed the same MoeMlpWeights fields.
            const packed_gu_w = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.weight");
            var sw_gate_w: mlx.mlx_array = undefined;
            var sw_gate_s: mlx.mlx_array = undefined;
            var sw_gate_b: mlx.mlx_array = undefined;
            var sw_up_w: mlx.mlx_array = undefined;
            var sw_up_s: mlx.mlx_array = undefined;
            var sw_up_b: mlx.mlx_array = undefined;
            var sw_down_w: mlx.mlx_array = undefined;
            var sw_down_s: mlx.mlx_array = undefined;
            var sw_down_b: mlx.mlx_array = undefined;
            if (packed_gu_w) |gu_w| {
                const w_pair = try splitPackedGateUp(gu_w, s);
                try owned_bf16.append(allocator, w_pair.gate);
                try owned_bf16.append(allocator, w_pair.up);
                sw_gate_w = w_pair.gate;
                sw_up_w = w_pair.up;
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.scales")) |gu_s| {
                    const s_pair = try splitPackedGateUp(gu_s, s);
                    try owned_bf16.append(allocator, s_pair.gate);
                    try owned_bf16.append(allocator, s_pair.up);
                    sw_gate_s = s_pair.gate;
                    sw_up_s = s_pair.up;
                } else {
                    sw_gate_s = mlx.mlx_array_new();
                    sw_up_s = mlx.mlx_array_new();
                }
                if (getLayerWeightOpt(weights, name_buf, prefix, li, "experts.gate_up_proj.biases")) |gu_b| {
                    const b_pair = try splitPackedGateUp(gu_b, s);
                    try owned_bf16.append(allocator, b_pair.gate);
                    try owned_bf16.append(allocator, b_pair.up);
                    sw_gate_b = b_pair.gate;
                    sw_up_b = b_pair.up;
                } else {
                    sw_gate_b = mlx.mlx_array_new();
                    sw_up_b = mlx.mlx_array_new();
                }
                sw_down_w = getLayerWeight(weights, name_buf, prefix, li, "experts.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.down_proj.biases") orelse mlx.mlx_array_new();
            } else {
                sw_gate_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.weight");
                sw_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.scales") orelse mlx.mlx_array_new();
                sw_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.biases") orelse mlx.mlx_array_new();
                sw_up_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.weight");
                sw_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.scales") orelse mlx.mlx_array_new();
                sw_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.biases") orelse mlx.mlx_array_new();
                sw_down_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.weight");
                sw_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.scales") orelse mlx.mlx_array_new();
                sw_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.biases") orelse mlx.mlx_array_new();
            }
            lw.mlp = .{
                .moe = .{
                    .router_w = getLayerWeight(weights, name_buf, prefix, li, "router.proj.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.biases") orelse mlx.mlx_array_new(),
                    .router_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.scale"),
                    .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.per_expert_scale"),
                    .switch_gate_w = sw_gate_w,
                    .switch_gate_s = sw_gate_s,
                    .switch_gate_b = sw_gate_b,
                    .switch_up_w = sw_up_w,
                    .switch_up_s = sw_up_s,
                    .switch_up_b = sw_up_b,
                    .switch_down_w = sw_down_w,
                    .switch_down_s = sw_down_s,
                    .switch_down_b = sw_down_b,
                    // Shared expert handled via lw.shared_mlp for Gemma 4 (separate branch in forward)
                    .shared_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
            // Pre-fold the sigma-MoE router norm scale: at runtime the router does
            // `rms_norm(x, router_scale * hidden_size^-0.5, eps)`. Multiplying once
            // at load time saves the per-layer multiply (3 ops × num_layers).
            if (lw.mlp.moe.router_scale) |rs| {
                const root_size: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(config.hidden_size)));
                const root_scalar = bf16Scalar(root_size, s);
                defer _ = mlx.mlx_array_free(root_scalar);
                var folded = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&folded, rs, root_scalar, s));
                try owned_bf16.append(allocator, folded);
                lw.mlp.moe.router_scale = folded;
            }
        } else if (layer_is_moe and is_laguna) {
            // Laguna: qwen3_moe WEIGHT NAMING (mlp.gate router bf16,
            // mlp.switch_mlp.* nvfp4 experts, mlp.shared_expert.* bf16 shared)
            // with hy_v3 ROUTING SEMANTICS — sigmoid + f32 selection bias
            // (mlp.gate.e_score_correction_bias) + top-k renorm + route_scale,
            // and the shared expert ALWAYS added, no gate (shared_ungated). The
            // nvfp4 experts carry scales (u8 fp8) but no biases (bias-less mode);
            // the router/shared weights are bf16 (null-ctx scales → pre-transposed
            // to plain matmul by maybeTransposeForBf16).
            lw.mlp = .{
                .moe = .{
                    .router_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                    .switch_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.weight"),
                    .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .switch_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.weight"),
                    .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .switch_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.weight"),
                    .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                    .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.e_score_correction_bias") orelse blk: {
                        // Some community conversions (mlx-lm quantizer) DROP the
                        // aux-loss-free bias — the reference zero-inits it, so
                        // selection reduces to plain sigmoid top-k. Non-null keeps
                        // moeMLP2 on the sigmoid chain (null falls back to softmax).
                        var zeros = mlx.mlx_array_new();
                        const zshape = [_]c_int{@intCast(config.num_experts)};
                        try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                        try owned_bf16.append(allocator, zeros);
                        break :blk zeros;
                    },
                    .route_norm = config.moe_route_norm,
                    .route_scale = config.router_scaling_factor,
                    .shared_ungated = true,
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
        } else if (layer_is_moe and is_hy3) {
            // Hy3 (hy_v3): stacked experts (already [E, out, packed] in MLX
            // conversions — never per-expert), the QUANTIZED router under
            // mlp.router.gate.* (8-bit on shipped checkpoints; scales optional
            // for a bf16 build — mlx-community oQ2e ships a bf16 router), the
            // f32 selection bias mlp.expert_bias, and an UNGATED shared expert
            // under mlp.shared_mlp.* (absent on 0-shared configs — binds empty
            // and moeMLP skips the branch). route_norm and router_scaling_factor
            // ride on the weights struct so moeMLP2 needs no config re-derivation
            // per call. The expert container name varies by converter
            // (`mlp.experts.*` in ox-ox builds, `mlp.switch_mlp.*` in mlx-lm
            // builds) — probe once and thread the resolved name through.
            const ex = hy3ExpertContainer(weights, name_buf, prefix, li);
            var exbuf: [64]u8 = undefined;
            lw.mlp = .{
                .moe = .{
                    .router_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.router.gate.weight"),
                    .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.scales") orelse mlx.mlx_array_new(),
                    .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.router.gate.biases") orelse mlx.mlx_array_new(),
                    .switch_gate_w = getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.weight")),
                    .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "gate_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_up_w = getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.weight")),
                    .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "up_proj.biases")) orelse mlx.mlx_array_new(),
                    .switch_down_w = getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.weight")),
                    .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.scales")) orelse mlx.mlx_array_new(),
                    .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, ex, "down_proj.biases")) orelse mlx.mlx_array_new(),
                    .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                    .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.weight") orelse mlx.mlx_array_new(),
                    .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                    .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                    .expert_bias = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.expert_bias") orelse blk: {
                        // moe_router_enable_expert_bias=false checkpoints ship no
                        // bias tensor; the reference keeps zeros — selection
                        // reduces to plain sigmoid top-k. Non-null keeps moeMLP2
                        // on the sigmoid chain (null would fall back to softmax).
                        var zeros = mlx.mlx_array_new();
                        const zshape = [_]c_int{@intCast(config.num_experts)};
                        try mlx.check(mlx.mlx_zeros(&zeros, &zshape, 1, .float32, s));
                        try owned_bf16.append(allocator, zeros);
                        break :blk zeros;
                    },
                    .route_norm = config.moe_route_norm,
                    .route_scale = config.router_scaling_factor,
                    .shared_ungated = true,
                },
            };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
            }
        } else if (layer_is_moe) {
            // Qwen3.5 MoE — also serves Qwen3-30B-A3B (`qwen3_moe`), which shares
            // this exact router/switch_mlp layout. Each `*_s`/`*_b` is loaded
            // optionally — Unsloth Dynamic checkpoints (e.g. Qwen3.6-A3B UD) leave
            // the router (`mlp.gate`) and the shared-expert gate
            // (`mlp.shared_expert_gate`) as plain bf16, with no scales/biases.
            // The shared-expert WEIGHTS themselves are also optional: qwen3_moe
            // (Qwen3-30B-A3B / Coder) dropped the shared expert entirely
            // (shared_expert_intermediate_size: 0, no mlp.shared_expert.*). When
            // absent they bind to empty handles and `shared_expert_gate_w` stays
            // null, which makes moeMLP early-return the routed-expert sum without
            // ever reading them — so no MISSING WEIGHT crash. The
            // `maybeTransposeForBf16` calls below pre-transpose bf16 weights from
            // `[out, in]` → `[in, out]` so `qmatmulBits` can dispatch to plain
            // `mlx_matmul`; they no-op on already-quantized AND on empty handles.
            lw.mlp = .{ .moe = .{
                .router_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
                .switch_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.weight"),
                .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .switch_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.weight"),
                .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .switch_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.weight"),
                .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
                .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
                .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
                .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_expert_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
                .shared_expert_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales") orelse mlx.mlx_array_new(),
                .shared_expert_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const mw = &lw.mlp.moe;
                try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, allocator, s);
                if (mw.shared_expert_gate_w) |*seg_w_ptr| {
                    try maybeTransposeForBf16(seg_w_ptr, mw.shared_expert_gate_s.?, &owned_bf16, allocator, s);
                }
            }
        } else {
            lw.mlp = .{ .dense = .{
                .gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight"),
                .gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
                .gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
                .up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight"),
                .up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array_new(),
                .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array_new(),
                .down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight"),
                .down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array_new(),
                .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array_new(),
            } };
            {
                const dw = &lw.mlp.dense;
                try maybeTransposeForBf16(&dw.gate_w, dw.gate_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.up_w, dw.up_s, &owned_bf16, allocator, s);
                try maybeTransposeForBf16(&dw.down_w, dw.down_s, &owned_bf16, allocator, s);
            }
        }

        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
    }

    return .{
        .moe_layers = moe_layers,
        .ssm_entries = ssm_entries,
        .owned_bf16 = try owned_bf16.toOwnedSlice(allocator),
    };
}

fn initHybridLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, _: mlx.mlx_stream) !struct { hybrid_layers: []HybridLayerWeights, ssm_entries: []SSMCacheEntry } {
    log.info("Precomputing hybrid layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const hybrid_layers = try allocator.alloc(HybridLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    const is_lfm2 = std.mem.eql(u8, config.model_type, "lfm2");
    const is_nemotron = std.mem.eql(u8, config.model_type, "nemotron_h");

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &hybrid_layers[i];
        const block_type = config.layer_block_types[i];

        // Input norm: LFM2 uses "operator_norm", Nemotron-H uses "norm"
        if (is_lfm2) {
            lw.input_norm = getLayerWeight(weights, name_buf, prefix, li, "operator_norm.weight");
        } else {
            lw.input_norm = getLayerWeight(weights, name_buf, prefix, li, "norm.weight");
        }

        // Post norm (before MLP): LFM2 uses "ffn_norm", Nemotron-H single-op blocks have none
        if (is_lfm2) {
            lw.post_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "ffn_norm.weight");
        } else {
            lw.post_norm = null;
        }

        // Initialize SSM/conv cache entry
        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };

        switch (block_type) {
            .gated_conv => {
                lw.op = .{ .gated_conv = .{
                    .in_proj_w = getLayerWeight(weights, name_buf, prefix, li, "conv.in_proj.weight"),
                    .in_proj_s = getLayerWeight(weights, name_buf, prefix, li, "conv.in_proj.scales"),
                    .in_proj_b = getLayerBias(weights, name_buf, prefix, li, "conv.in_proj.biases", &config),
                    .conv_w = getLayerWeight(weights, name_buf, prefix, li, "conv.conv.weight"),
                    .out_proj_w = getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.weight"),
                    .out_proj_s = getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.scales"),
                    .out_proj_b = getLayerBias(weights, name_buf, prefix, li, "conv.out_proj.biases", &config),
                } };
            },
            .attention => {
                if (is_nemotron) {
                    // Nemotron-H: mixer.{q,k,v,o}_proj, no QK norms
                    // Use Opt for biases — mxfp8 quantized layers may lack them
                    lw.op = .{ .full_attn = .{
                        .q_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.q_proj.weight"),
                        .q_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.q_proj.scales"),
                        .q_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.q_proj.biases") orelse mlx.mlx_array_new(),
                        .k_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.k_proj.weight"),
                        .k_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.k_proj.scales"),
                        .k_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.k_proj.biases") orelse mlx.mlx_array_new(),
                        .v_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.v_proj.weight"),
                        .v_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.v_proj.scales"),
                        .v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.v_proj.biases") orelse mlx.mlx_array_new(),
                        .o_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.o_proj.weight"),
                        .o_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.o_proj.scales"),
                        .o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.o_proj.biases") orelse mlx.mlx_array_new(),
                        .q_norm = mlx.mlx_array_new(),
                        .k_norm = mlx.mlx_array_new(),
                    } };
                } else {
                    // LFM2: self_attn.{q,k,v}_proj + out_proj, QK layernorms
                    lw.op = .{ .full_attn = .{
                        .q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                        .q_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.scales"),
                        .q_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                        .k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight"),
                        .k_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.scales"),
                        .k_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config),
                        .v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight"),
                        .v_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.scales"),
                        .v_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", &config),
                        .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.weight"),
                        .o_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.scales"),
                        .o_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.out_proj.biases", &config),
                        .q_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_layernorm.weight") orelse mlx.mlx_array_new(),
                        .k_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_layernorm.weight") orelse mlx.mlx_array_new(),
                    } };
                }
            },
            .mamba2 => {
                lw.op = .{ .mamba2 = .{
                    .in_proj_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.in_proj.weight"),
                    .in_proj_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.in_proj.scales"),
                    .in_proj_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.in_proj.biases") orelse mlx.mlx_array_new(),
                    .conv1d_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.conv1d.weight"),
                    .conv1d_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.conv1d.bias"),
                    .A_log = getLayerWeight(weights, name_buf, prefix, li, "mixer.A_log"),
                    .D = getLayerWeight(weights, name_buf, prefix, li, "mixer.D"),
                    .dt_bias = getLayerWeight(weights, name_buf, prefix, li, "mixer.dt_bias"),
                    .norm_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.norm.weight"),
                    .out_proj_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.out_proj.weight"),
                    .out_proj_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.out_proj.scales"),
                    .out_proj_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.out_proj.biases") orelse mlx.mlx_array_new(),
                } };
            },
            .mlp => {
                // Nemotron-H standalone MLP (ReLU^2, ungated: up + down only)
                lw.op = .{ .simple_mlp = .{
                    .up_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.up_proj.weight"),
                    .up_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.up_proj.scales"),
                    .up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.up_proj.biases") orelse mlx.mlx_array_new(),
                    .down_w = getLayerWeight(weights, name_buf, prefix, li, "mixer.down_proj.weight"),
                    .down_s = getLayerWeight(weights, name_buf, prefix, li, "mixer.down_proj.scales"),
                    .down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mixer.down_proj.biases") orelse mlx.mlx_array_new(),
                } };
            },
            .moe => {
                // TODO: Nemotron MoE support
                unreachable;
            },
        }

        // MLP: present for all LFM2 layers, absent for Nemotron-H single-op blocks
        if (is_lfm2) {
            // LFM2 uses feed_forward.w1 (gate), w3 (up), w2 (down) — SwiGLU
            lw.mlp = .{
                .gate_w = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w1.weight"),
                .gate_s = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w1.scales"),
                .gate_b = getLayerBias(weights, name_buf, prefix, li, "feed_forward.w1.biases", &config),
                .up_w = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w3.weight"),
                .up_s = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w3.scales"),
                .up_b = getLayerBias(weights, name_buf, prefix, li, "feed_forward.w3.biases", &config),
                .down_w = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w2.weight"),
                .down_s = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w2.scales"),
                .down_b = getLayerBias(weights, name_buf, prefix, li, "feed_forward.w2.biases", &config),
            };
        } else {
            lw.mlp = null;
        }
    }

    return .{ .hybrid_layers = hybrid_layers, .ssm_entries = ssm_entries };
}

fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (comptime structFields(FullAttnWeights)) |field| {
        // Dense bf16 full-attn layers carry null-ctx scales/biases — skip those
        // so null arrays don't poison the eval batch. Mirrors the linear/mlp paths.
        const arr = @field(fa, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (comptime structFields(LinearAttnWeights)) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        const za_field = comptime std.mem.startsWith(u8, field.name, "z_") or std.mem.startsWith(u8, field.name, "a_");
        const skip_za = za_field and la.combined_proj;
        if (!skip_za) {
            const arr = @field(la, field.name);
            // Plain-bf16 layers (Unsloth Dynamic) carry null scales/biases — skip those.
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        }
    }
}

fn appendHybridMlpWeights(vec: mlx.mlx_vector_array, hw: *const HybridMlpWeights) void {
    // Plain-bf16 layers (Unsloth Dynamic) carry null-ctx scales/biases — skip those
    // so they don't pollute the eval batch. Mirrors `appendLinearAttnWeights`.
    switch (hw.*) {
        .moe => |*mw| {
            inline for (comptime structFields(MoeMlpWeights)) |field| {
                if (field.type == ?mlx.mlx_array) {
                    if (@field(mw, field.name)) |arr| {
                        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                } else if (field.type == mlx.mlx_array) {
                    const arr = @field(mw, field.name);
                    if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                }
            }
        },
        .dense => |*dw| {
            inline for (comptime structFields(DenseMlpWeights)) |field| {
                const arr = @field(dw, field.name);
                if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        },
    }
}

// ── Utility functions ──

/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

/// Last-axis size of an array, as the input-dimension hint for
/// `quantParamsHinted`. Null for null-ctx handles or degenerate shapes.
inline fn lastDim(x: mlx.mlx_array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0) return null;
    const d = shape[shape.len - 1];
    return if (d > 0) @as(u32, @intCast(d)) else null;
}

/// Detect a quantized weight's (bits, group_size, mode) from its scales tensor
/// plus the model config:
/// - uint8 scales → fp8-family. In an nvfp4/mxfp4/mxfp8 model the config's
///   mode/bits/group_size apply (the fp8 scheme is uniform within a model).
///   Under an affine config this is the legacy "mxfp8 tensor inside an affine
///   checkpoint" case: bits is always 8, group size from the shape ratio.
/// - float scales → affine. In an affine model, bits via detectQuantBits
///   against the config group_size (mixed-precision models share one gs). For
///   affine OVERRIDES inside a non-affine model (e.g. nvfp4 QAT checkpoints
///   keep the shared MLP at affine 8-bit/gs64) the config group_size doesn't
///   apply: w_cols*32/s_cols only pins bits×gs, so solve exactly with the
///   caller's `in_dim` (activation inner dim); without a hint, assume mlx-lm's
///   override default group size 64.
/// Solve an affine weight's exact (bits, group_size) from packed-column
/// geometry when the input dimension is known:
///   bits = w_cols * 32 / in_dim,  group_size = in_dim / s_cols
/// Returns null unless both divide EXACTLY and land on values MLX's affine
/// packing supports — callers then fall back to config-based detection.
/// This is what lets a sidecar (e.g. an MTP head quantized 5-bit/gs-128 over
/// a 4-bit/gs-64 trunk) resolve per-weight instead of inheriting the trunk's
/// group size.
/// True when an embedding/lm_head TABLE is stored dense (float dtype) even
/// though the config declares global quantization. Mixed checkpoints quantize
/// per-tensor — hy_v3 2-bit ships a bf16 embed_tokens beside 2-bit experts —
/// so scales-presence must be decided by the WEIGHT's dtype, never the
/// config's bits (a packed uint32 table missing scales still crashes
/// honestly at the mandatory fetch).
pub fn floatDtypeTable(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float16 or dtype == .float32;
}

test "floatDtypeTable: float tables are dense, packed/quantized ones are not" {
    try std.testing.expect(floatDtypeTable(.bfloat16));
    try std.testing.expect(floatDtypeTable(.float16));
    try std.testing.expect(floatDtypeTable(.float32));
    try std.testing.expect(!floatDtypeTable(.uint32)); // packed affine quant
    try std.testing.expect(!floatDtypeTable(.uint8)); // fp8-encoded scales/modes
    try std.testing.expect(!floatDtypeTable(.int32));
}

/// Additive symmetric band mask for bidirectional sliding-window layers:
/// [1, 1, L, L] bf16, 0 where |i-j| < window, -inf outside — the causal
/// sliding semantic (lookback window-1 + self) mirrored in both directions.
/// Broadcasts additively against a [B, 1, 1, L] key-pad mask.
pub fn encoderBandMask(allocator: std.mem.Allocator, seq_len: usize, window: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(buf);
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            const d = if (i > j) i - j else j - i;
            buf[i * seq_len + j] = if (d < window) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(seq_len), @intCast(seq_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

test "encoderBandMask: symmetric |i-j| < window band, -inf outside" {
    const s = mlx.gpuStream();
    const mask = try encoderBandMask(testing.allocator, 4, 2, s);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const vals = mlx.mlx_array_data_float32(f).?;
    // Row 0: [0, 0, -inf, -inf]; row 2: [-inf, 0, 0, 0].
    try testing.expectEqual(@as(f32, 0), vals[0]);
    try testing.expectEqual(@as(f32, 0), vals[1]);
    try testing.expect(std.math.isNegativeInf(vals[2]));
    try testing.expect(std.math.isNegativeInf(vals[3]));
    try testing.expect(std.math.isNegativeInf(vals[2 * 4 + 0]));
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 1]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 2]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 3]);
    const shape = mlx.getShape(mask);
    try testing.expectEqual(@as(c_int, 1), shape[0]);
    try testing.expectEqual(@as(c_int, 1), shape[1]);
    try testing.expectEqual(@as(c_int, 4), shape[2]);
    try testing.expectEqual(@as(c_int, 4), shape[3]);
}

pub fn affineParamsFromGeometry(w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: u32) ?QuantParams {
    if (sc.ctx == null or in_dim == 0) return null;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (w_cols == 0 or s_cols == 0) return null;
    if ((w_cols * 32) % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = (w_cols * 32) / in_dim;
    const gs = in_dim / s_cols;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    switch (gs) {
        32, 64, 128 => {},
        else => return null,
    }
    return .{ .bits = bits, .group_size = gs, .mode = .affine };
}

pub fn computeQuantParams(config: *const ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;

    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (config.quant_mode != .affine) {
            return .{ .bits = config.quant_bits, .group_size = config.quant_group_size, .mode = config.quant_mode };
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }

    if (config.quant_mode == .affine) {
        // Exact per-weight solve when the caller knows the input dim — a
        // hinted tensor whose geometry contradicts the config's group size
        // (MTP sidecars mix gs 128/64/32 over a gs-64 trunk) resolves to its
        // TRUE params. Consistent tensors solve to the config values anyway.
        if (in_dim) |in| {
            if (affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        }
        return .{
            .bits = detectQuantBits(w, sc, config.quant_group_size),
            .group_size = config.quant_group_size,
            .mode = .affine,
        };
    }

    if (in_dim) |in| {
        if (in > 0 and s_cols > 0 and w_cols > 0) {
            return .{ .bits = (w_cols * 32) / in, .group_size = in / s_cols, .mode = .affine };
        }
    }
    if (w_cols > 0 and s_cols > 0) {
        const ratio = (w_cols * 32) / s_cols; // = bits × group_size
        return .{ .bits = @max(ratio / 64, 1), .group_size = 64, .mode = .affine };
    }
    return .{ .bits = 8, .group_size = 64, .mode = .affine };
}

/// MoE routing chain (negate→argpartition→slice→softmax→take→sum→expand→divide).
/// Free-function variant of `Transformer.moeRoutingUncompiled` so unit tests can
/// exercise the pure subgraph without constructing a full Transformer. Returns
/// owned `inds` (int32, [..., k]) and `norm_scores` (bf16, [..., k]) — caller
/// must free both.
/// GatedDeltaNet gating chain: g = exp(-exp(A_log) * softplus(a + dt_bias)),
/// computed in float32 for stability and returned as bfloat16. Mirrors
/// mlx-lm's `compute_g` (which is `@mx.compile`d). Pure — serves as both the
/// compiled-closure body and the uncompiled fallback. Returns owned array.
fn gdnGateChain(A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var A_log_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_log_f32);
    try mlx.check(mlx.mlx_astype(&A_log_f32, A_log, .float32, s));
    var exp_A = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_A);
    try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, s));

    // softplus(a + dt_bias) = log1p(exp(a + dt_bias))
    var a_plus_dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_plus_dt);
    try mlx.check(mlx.mlx_add(&a_plus_dt, a, dt_bias, s));
    var a_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_f32);
    try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, s));
    var exp_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_a);
    try mlx.check(mlx.mlx_exp(&exp_a, a_f32, s));
    var sp_inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp_inner);
    try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, s));

    var neg_decay = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_decay);
    try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, s));
    var neg_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_neg);
    try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, s));
    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, s));
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g_f32, .bfloat16, s));
    return g;
}

fn moeRoutingChain(router_logits: mlx.mlx_array, k: c_int, s: mlx.mlx_stream) !Transformer.MoeRouting {
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, router_logits, s));

    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg_logits, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, router_logits, -1, true, s));

    var top_weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_weights);
    try mlx.check(mlx.mlx_take_along_axis(&top_weights, probs, inds, -1, s));

    var weight_sum_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum_raw);
    try mlx.check(mlx.mlx_sum_axis(&weight_sum_raw, top_weights, -1, false, s));
    var weight_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum);
    try mlx.check(mlx.mlx_expand_dims(&weight_sum, weight_sum_raw, -1, s));
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_divide(&norm_scores, top_weights, weight_sum, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

/// HF `_compute_yarn_parameters` → the per-dim RoPE DENOMINATOR array (`freqs`)
/// that mlx_fast_rope consumes: mlx computes angle = position / freqs[i], so we
/// return 1/inv_freq where inv_freq is the YaRN-corrected inverse frequency.
/// `out.len` must equal the rotary half-dim (= int(head_dim * partial) / 2).
/// Pure f64 math (no MLX) so it unit-tests directly against reference values.
/// Mirrors modeling_laguna.py's rope for the full-attention layers verbatim,
/// with truncate=True (Laguna's rope_parameters omits the flag → default True).
fn computeYarnFreqs(
    out: []f64,
    head_dim: u32,
    partial: f32,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max: u32,
) void {
    const dim: f64 = @floor(@as(f64, @floatFromInt(head_dim)) * @as(f64, partial)); // 64
    const n = out.len; // dim/2 = 32
    const mp: f64 = @floatFromInt(orig_max);
    const two_log_base = 2.0 * @log(base);
    // find_correction_dim(num_rotations) = dim * ln(orig_max/(num_rot*2π)) / (2·ln base)
    const corr = struct {
        fn f(num_rot: f64, d: f64, max_pos: f64, tlb: f64) f64 {
            return (d * @log(max_pos / (num_rot * 2.0 * std.math.pi))) / tlb;
        }
    }.f;
    var low = @floor(corr(beta_fast, dim, mp, two_log_base));
    var high = @ceil(corr(beta_slow, dim, mp, two_log_base));
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    var ramp_denom = high - low;
    if (ramp_denom == 0) ramp_denom = 0.001;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        // pos_freqs = base^(arange(0,dim,2)[i]/dim) = base^(2i/dim)
        const pos_freq = std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / dim);
        const inv_extrapolation = 1.0 / pos_freq;
        const inv_interpolation = 1.0 / (factor * pos_freq);
        // linear_ramp_factor(low, high) clamped to [0,1], then extrapolation factor = 1 - ramp
        var ramp = (@as(f64, @floatFromInt(i)) - low) / ramp_denom;
        if (ramp < 0) ramp = 0;
        if (ramp > 1) ramp = 1;
        const extrapolation_factor = 1.0 - ramp;
        const inv_freq = inv_interpolation * (1.0 - extrapolation_factor) + inv_extrapolation * extrapolation_factor;
        out[i] = 1.0 / inv_freq;
    }
}

/// Hy3 (hy_v3 / DeepSeek-V3-style) sigmoid routing chain — mirrors the
/// reference `expert_select` (hy_v3.py):
///   scores = sigmoid(logits) in FLOAT32 (the fp32-router class: a bf16
///   sigmoid+bias flips near-tie expert picks on real checkpoints);
///   top-k SELECTED on scores + expert_bias, WEIGHTED by the unbiased scores;
///   if route_norm and k > 1: weights /= (sum + 1e-20);
///   weights *= route_scale; cast bf16 for the expert-combine multiply
///   (shipped checkpoints run enable_moe_fp32_combine = false).
/// Returns owned `inds` + `norm_scores` — caller must free both.
fn hy3RoutingChain(router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array, k: c_int, route_norm: bool, route_scale: f32, s: mlx.mlx_stream) !Transformer.MoeRouting {
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits_f32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, scores, expert_bias, s));

    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, biased, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    // Weights = ORIGINAL (unbiased) sigmoid scores at the selected indices.
    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_take_along_axis(&top, scores, inds, -1, s));

    var weights_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights_f32);
    if (route_norm and k > 1) {
        var sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&sum_raw, top, -1, true, s));
        const eps = mlx.mlx_array_new_float(1e-20);
        defer _ = mlx.mlx_array_free(eps);
        var sum_eps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_eps);
        try mlx.check(mlx.mlx_add(&sum_eps, sum_raw, eps, s));
        try mlx.check(mlx.mlx_divide(&weights_f32, top, sum_eps, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&weights_f32, top));
    }

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (route_scale != 1.0) {
        const scale_arr = mlx.mlx_array_new_float(route_scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        try mlx.check(mlx.mlx_multiply(&scaled, weights_f32, scale_arr, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&scaled, weights_f32));
    }

    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_astype(&norm_scores, scaled, .bfloat16, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

// ── Decode sub-block profiler (MLX_SERVE_DECODE_PROFILE=1) ──
// Serializes the S=1 decode forward with per-sub-block evals to attribute the
// per-token GPU cost across embed / attn / mlp / lm_head. DIAGNOSTIC ONLY: the
// evals defeat the async pipeline (absolute tok/s drops while active), so the
// signal is the RELATIVE split plus the serialized total vs the pipelined
// per-token time — a serialized total near the pipelined per-token means we are
// compute-bound; far below it means the gap is dispatch/pipeline overhead.
const DecodeProf = struct {
    calls: u64 = 0,
    embed_ns: u64 = 0,
    attn_ns: u64 = 0,
    mlp_ns: u64 = 0,
    lmhead_ns: u64 = 0,
    // MoE internals (summed across all MoE layers per token)
    moe_router_ns: u64 = 0,
    moe_experts_ns: u64 = 0,
    moe_shared_ns: u64 = 0,
};
var decode_prof: DecodeProf = .{};
var decode_prof_enabled: ?bool = null;

// Self-contained monotonic lap timer (this Zig nightly has no std.time.Timer;
// the repo times via std.Io). `lap()` returns ns since the previous lap.
const ProfClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    fn init() ProfClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    fn lap(self: *ProfClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};

fn decodeProfileEnabled() bool {
    if (decode_prof_enabled) |v| return v;
    const v = std.c.getenv("MLX_SERVE_DECODE_PROFILE") != null;
    decode_prof_enabled = v;
    return v;
}

// MoE decode expert compute: our self-built libmlx serializes `gather_qmm` at
// decode width (~10× vs a batched `quantized_matmul`; plain qmv overlaps fine —
// verified by the MoE gather µbench). So at S==1 we CAN compute the top-K experts
// as `take(experts) + broadcast(x) + batched quantized_matmul` (batchedExpertMm)
// instead of one gather_qmm. Numerically the same per-expert matmul; not bit-
// identical to the gather kernel (different reduction order, qmv-vs-qmm class).
//
// But this is a per-arch VALIDATED opt-in, NOT default-on-for-all-MoE. The
// take-materialization (extracting the top-K experts from the bank every token)
// only pays off where the serialized gather dominates — Laguna 2-bit large
// experts: 17→48 tok/s. On small-expert MoEs the materialization overhead is a
// NET LOSS: gemma4-26B-A4B measured 114→85 and Qwen3.6-MoE regressed too when
// this path was briefly default-on (the 26.7.10 bench). Same class as the
// "eligibility predicate silently adopts every matching shape" kernel rule.
//
// Policy (pure, unit-tested by `batchedExpertDecodePolicy` test):
//   - MLX_SERVE_MOE_GATHER_DECODE=1  → hard force gather (beats everything).
//   - MLX_SERVE_MOE_BATCHED_DECODE=1 → hard force batched (for A/B on any arch).
//   - otherwise                      → batched ONLY for laguna, gather for the rest.
fn batchedExpertDecodePolicy(model_type: []const u8, gather_force: bool, batched_force: bool) bool {
    if (gather_force) return false;
    if (batched_force) return true;
    return std.mem.eql(u8, model_type, "laguna");
}

var moe_gather_force_env: ?bool = null;
var moe_batched_force_env: ?bool = null;
fn envFlagCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = std.c.getenv(name) != null;
    cache.* = v;
    return v;
}
fn useBatchedExpertDecode(self: *const Transformer) bool {
    return batchedExpertDecodePolicy(
        self.config.model_type,
        envFlagCached(&moe_gather_force_env, "MLX_SERVE_MOE_GATHER_DECODE"),
        envFlagCached(&moe_batched_force_env, "MLX_SERVE_MOE_BATCHED_DECODE"),
    );
}

fn decodeProfReport() void {
    const n = decode_prof.calls;
    if (n == 0) return;
    const per = struct {
        fn ms(ns: u64, calls: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(calls)) / 1.0e6;
        }
    };
    const e = per.ms(decode_prof.embed_ns, n);
    const a = per.ms(decode_prof.attn_ns, n);
    const m = per.ms(decode_prof.mlp_ns, n);
    const l = per.ms(decode_prof.lmhead_ns, n);
    const mr = per.ms(decode_prof.moe_router_ns, n);
    const mx_ = per.ms(decode_prof.moe_experts_ns, n);
    const ms_ = per.ms(decode_prof.moe_shared_ns, n);
    log.info("[decode-prof] n={d} serial/tok={d:.2}ms  embed={d:.2} attn={d:.2} mlp={d:.2} lmhead={d:.2} ms | moe: router={d:.2} experts={d:.2} shared={d:.2} ms\n", .{ n, e + a + m + l, e, a, m, l, mr, mx_, ms_ });
    decode_prof = .{};
}

// ── MoE decode gather-qmv kernel ──
// MLX's `gather_qmm` at decode width costs O(EXPERT BANK SIZE), not O(work), in
// any C/C++ process (it is flat under pip-Python with the identical dylib —
// measured E-sweep in the `MoE decode gather` µbench: 30/50/77/300 us at banks
// of 8/25/50/201 MB, all reading the SAME 10 experts). 141 gathers/token x 300 us
// is the whole 42 ms/token Laguna decode. `take` from the same 201 MB bank does
// NOT pay it, so indexed reads are fine; the penalty is specific to gather_qmm's
// addressing.
//
// The shipped workaround (`batchedExpertMm`: take + broadcast + batched qmm)
// dodges it but pays a materialization round-trip — read 9.8 MB of experts,
// write 9.8 MB, read 9.8 MB back = 3x the ideal traffic, which is exactly its
// measured 70 us against a 67 us bandwidth prediction. It cannot go faster.
//
// This kernel reads the bank IN PLACE with GPU-resident indices: one simdgroup
// per (expert slot, output row), each lane striding the packed row and reducing
// with simd_sum. Ideal traffic, no copy, no host sync for the top-K indices.
//
// `X_PER_EXPERT` picks the input layout: 0 = one shared x [K] (gate/up
// projections, every expert sees the same token), 1 = x [TOPK, K] (the down
// projection, where each expert consumes its own activation).
fn gatherQmvSource(comptime x_per_expert: bool) [:0]const u8 {
    // Everything is indexed off the raw buffer names rather than through
    // explicitly address-spaced pointers: MLX generates the signature from each
    // input's ACTUAL dtype and puts arrays with fewer than 8 elements in
    // `constant` instead of `device` (backend/common/metal_kernel.cpp), so a
    // hand-declared `const device T*` would mismatch on both counts. `x` in
    // particular arrives as float32 on Laguna while scales/biases are bf16.
    //
    // A uint4 (128-bit) load variant was measured and is NOT kept: 38.9 us vs
    // 37.3 for the scalar form. At ~5 ALU ops per dequantized value this loop
    // is ALU-bound, not load-width bound, so widening the loads only adds code.
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;      // output row within the expert
        \\uint e = thread_position_in_grid.z;      // top-K slot
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\int VPW = 32 / BITS;                     // quantized values per uint32 word
        \\int K_by_p = K / VPW;                    // packed words per row
        \\int K_by_gs = K / GS;                    // quant groups per row
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[e];                      // bank row for this slot
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xoff = {s};
        \\
        \\// Four independent accumulators so the per-lane FMA chain is not one
        \\// serial dependency; VPW is 16/8/4 for bits 2/4/8, always a multiple of 4.
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
        \\  uint32_t packed = w_q[wbase + (size_t)pack];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;                  // GS >= VPW, so one group per word
        \\  float sj = float(scales[gbase + (size_t)gi]);
        \\  float bj = float(biases[gbase + (size_t)gi]);
        \\  for (int ki = 0; ki < VPW; ki += 4) {{
        \\    size_t xi = xoff + (size_t)(k_base + ki);
        \\    uint32_t q = packed >> (ki * BITS);
        \\    a0 += float(x[xi + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
        \\    a1 += float(x[xi + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
        \\    a2 += float(x[xi + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
        \\    a3 += float(x[xi + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
        \\  }}
        \\}}
        \\float acc = simd_sum((a0 + a1) + (a2 + a3));
        \\if (lane == 0) {{
        \\  y[(size_t)e * (size_t)N + (size_t)n] = T(acc);
        \\}}
    , .{if (x_per_expert) "(size_t)e * (size_t)K" else "0"});
}

const GQMV_SOURCES = [2][:0]const u8{ gatherQmvSource(false), gatherQmvSource(true) };
const GQMV_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv", "mlxserve_moe_gather_qmv_px" };
var gqmv_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvKernel(x_per_expert: bool) !mlx.mlx_fast_metal_kernel {
    const idx: usize = @intFromBool(x_per_expert);
    if (gqmv_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_NAMES[idx],
        in_vec,
        out_vec,
        GQMV_SOURCES[idx],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_kernels[idx] = kernel;
    return kernel;
}

var gqmv_disabled_env: ?bool = null;
var gqmv_engaged: bool = false;

/// Decode-width gathered qmv over an expert bank, read in place.
/// `x`: [K] shared (x_per_expert=false) or [TOPK, K] (true), bf16/fp16.
/// `w`/`sc`/`bi`: the affine bank, [E, N, K*BITS/32] / [E, N, K/GS].
/// `inds`: [TOPK] uint32 bank rows. Returns [TOPK, N], or null when the shape
/// or quant geometry is outside the supported set (caller falls back).
fn gatherQmv(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
) !?mlx.mlx_array {
    if (envFlagCached(&gqmv_disabled_env, "MLX_SERVE_MOE_GATHER_QMV_OFF")) return null;
    // The kernel implements the AFFINE dequant (q * scale + bias) only. Laguna's
    // upstream checkpoint ships nvfp4 experts, so this guard is load-bearing:
    // running an nvfp4 bank through affine math is silently wrong, not a crash.
    if (mode != .affine) return null;
    // Only widths where one uint32 holds a whole number of values (5/6-bit
    // affine is byte-packed differently — those fall back).
    if (bits != 2 and bits != 4 and bits != 8) return null;
    if (group_size % (32 / bits) != 0) return null; // one quant group per word
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    const want: i64 = if (x_per_expert) @as(i64, topk) * K else K;
    if (xelems != want) return null;

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ topk, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    // One simdgroup per (slot, row); 8 simdgroups to a threadgroup.
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, topk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvKernel(x_per_expert);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

/// Gathered matmul for MoE expert dispatch — handles both quantized and dense bf16.
/// Quantized (sc.ctx != null): mlx_gather_qmm with transpose_w=true, w stored [E,out,in].
/// Dense bf16 (sc.ctx == null): mlx_gather_mm; w was pre-transposed to [E,in,out] at load
/// (maybeTransposeForBf16 + generalized transposeBf16Weight), so x @ w is correct with no
/// transpose flag — mirrors mlx-lm's `gather_mm(x, weight.swapaxes(-1,-2))`.
fn gatherExpertMm(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, sorted: bool, s: mlx.mlx_stream) !void {
    if (sc.ctx == null) {
        // mlx 0.31.2's `mlx_gather_mm` returns WRONG results with sorted_indices=true
        // for the dense (non-quantized) path — the quantized `mlx_gather_qmm` honors
        // the flag correctly, but the dense kernel does not. The sorted-indices flag is
        // only a performance hint (rhs_indices ARE sorted here), so forcing false is
        // always numerically correct; it just forgoes the sorted-stream optimization.
        // Without this, dense bf16 MoE prefill (the sorted path, S>1) produced fluent
        // but semantically-wrong output (e.g. Qwen3.6-35B-A3B-bf16 calling clean prompts
        // "jumbled"). Verified byte-identical to mlx-lm once forced false. `sorted` is
        // still honored by the quantized branch below, where gather_qmm handles it
        // correctly.
        try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_idx, rhs_idx, false, s));
    } else {
        try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, bi, lhs_idx, rhs_idx, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), sorted, s));
    }
}

/// DiffusionGemma packs each expert's gate and up projections in one tensor
/// `experts.gate_up_proj` of shape [E, 2*M, X], gate rows first (HF chunks
/// the projected output in halves: gate = [..., :M], up = [..., M:]).
/// Slicing axis 1 maps the packed layout onto MoeMlpWeights' separate
/// switch_gate_*/switch_up_* fields with zero forward-pass changes; the same
/// split serves the packed-u32 weight, its scales, and its biases (all share
/// the [E, rows, X] expert-output layout — quant groups run along axis 2).
/// Caller owns both returned arrays.
fn splitPackedGateUp(arr: mlx.mlx_array, s: mlx.mlx_stream) !struct { gate: mlx.mlx_array, up: mlx.mlx_array } {
    const shape = mlx.getShape(arr);
    if (shape.len != 3) return error.BadPackedGateUpShape;
    const two_m = shape[1];
    const m = @divExact(two_m, 2);
    const strides = [_]c_int{ 1, 1, 1 };

    var gate_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_view);
    const g_start = [_]c_int{ 0, 0, 0 };
    const g_stop = [_]c_int{ shape[0], m, shape[2] };
    try mlx.check(mlx.mlx_slice(&gate_view, arr, &g_start, 3, &g_stop, 3, &strides, 3, s));

    var up_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_view);
    const u_start = [_]c_int{ 0, m, 0 };
    const u_stop = [_]c_int{ shape[0], two_m, shape[2] };
    try mlx.check(mlx.mlx_slice(&up_view, arr, &u_start, 3, &u_stop, 3, &strides, 3, s));

    // gather_qmm reads the expert weight buffers as ROW-CONTIGUOUS — a lazy
    // slice view has parent strides and silently produces zeros. Materialize
    // both halves (one-time cost at load).
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_contiguous(&gate, gate_view, false, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_contiguous(&up, up_view, false, s));

    return .{ .gate = gate, .up = up };
}

// ── Prefill-width dequant+GEMM qmm route ──
// Stock MLX qmm_t runs a 32x32x32 steel tile; at prefill widths (M >= 2048)
// on the qwen 27B shapes that underuses the compute units by ~10% (µbench,
// M4 Max: gate/up q4 M=2048 stock 28.05 ms vs dequant+bf16-GEMM 25.36 ms
// INCLUDING the per-call dequant; pre-dequantized floor 24.64 ms). oMLX
// closes the same gap by re-instantiating qmm_t at bigger tiles (bm 64-128,
// their qwen35_q4_mlp patch); we get within ~3% of that with zero custom
// kernel: dequantize the weight to a bf16 [N,K] transient and run MLX's
// steel GEMM. The transient repeats identical sizes per layer, so the MLX
// allocator cache recycles it; numerics differ from in-kernel dequant only
// by the bf16 rounding of w = s*q + b (pinned no-worse-than-stock by test).
// Decode (M=1) and spec-verify widths never route. Kill switch:
// MLX_SERVE_PREFILL_DQ_GEMM=0.
pub const PREFILL_DQ_GEMM_MIN_M: usize = 2048;

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;
var prefill_dq_gemm_env_cached: ?bool = null;

pub fn prefillDqGemmEnabled() bool {
    if (prefill_dq_gemm_override) |v| return v;
    if (prefill_dq_gemm_env_cached) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_PREFILL_DQ_GEMM");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    prefill_dq_gemm_env_cached = enabled;
    return enabled;
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var prefill_dq_gemm_engaged: u64 = 0;

fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !mlx.mlx_array {
    // Plain BF16 weight: scales array is unset. Used by mixed-precision Unsloth
    // Dynamic checkpoints that leave a subset of layers (e.g. linear_attn
    // projections in Qwen3.6 UD) unquantized. The weight is pre-transposed at
    // load to [in, out] so a single mlx_matmul does the contraction.
    if (sc.ctx == null) {
        var fp_result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&fp_result, x, w, s));
        return fp_result;
    }

    // Non-affine model-wide mode (nvfp4 / mxfp4 / mxfp8 from config.json):
    // bias-less by construction; bits/group_size come from the config like
    // affine. Checked BEFORE the legacy null-bias heuristic below, which
    // would misroute e.g. an nvfp4 weight to mxfp8.
    if (mode != .affine) {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            mlx.mlx_array{ .ctx = null },
            true,
            mlx.mlx_optional_int.some(@intCast(group_size)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            mode.cstr(),
            s,
        ));
        return result;
    }

    // Legacy heuristic for mixed checkpoints whose config declares affine but
    // whose individual tensors lack biases (NVIDIA mxfp8 layers): biases array
    // has null ctx (created with mlx_array_new()).
    const is_mxfp = bi.ctx == null;

    if (is_mxfp) {
        // mxfp8: bits is always 8; infer group_size from scales/weight shape ratio
        const mxfp_bits: u32 = 8;
        const s_shape = mlx.getShape(sc);
        const w_shape = mlx.getShape(w);
        const mxfp_gs: u32 = if (s_shape.len >= 2 and w_shape.len >= 2) blk: {
            const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
            const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
            if (s_cols > 0) break :blk (w_cols * 32) / (s_cols * mxfp_bits);
            break :blk 32;
        } else 32;

        const null_bi = mlx.mlx_array{ .ctx = null };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            null_bi,
            true,
            mlx.mlx_optional_int.some(@intCast(mxfp_gs)),
            mlx.mlx_optional_int.some(@intCast(mxfp_bits)),
            "mxfp8",
            s,
        ));
        return result;
    }

    // Spec-verify fast path: M=2..6-row activations (verify forwards, small
    // decode batches) route through the split-K verify kernel when eligible
    // (see VERIFY_QMM_SOURCE) — stock qmm's qmv/steel dead zone.
    if (try verifyQmm(s, x, w, sc, bi, bits, group_size)) |vy| return vy;

    // Prefill-width fast path: M >= 2048 dequantizes + runs the steel bf16
    // GEMM (see prefillDqGemm) — stock qmm_t's 32x32x32 tile dead zone.
    if (try prefillDqGemm(x, w, sc, bi, bits, group_size, s)) |py| return py;

    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(
        &result,
        x,
        w,
        sc,
        bi,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return result;
}

/// The dequant+GEMM prefill route (see the block comment above
/// prefillDqGemmEnabled). Returns null when the call must stay on stock qmm.
fn prefillDqGemm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!prefillDqGemmEnabled()) return null;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    const last = lastDim(x) orelse return null;
    if (last == 0) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(last));
    if (rows < PREFILL_DQ_GEMM_MIN_M) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;

    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = xd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, dq_t, s));
    prefill_dq_gemm_engaged += 1;
    return out;
}

/// Extract timestep t from a [B, T, H, D] tensor → [B, H, D]
fn sliceTimestep4(arr: mlx.mlx_array, batch: c_int, heads: c_int, dim: c_int, t: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, t, 0, 0 };
    const stop = [_]c_int{ batch, t + 1, heads, dim };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, arr, &start, 4, &stop, 4, &strides, 4, s));
    const out_shape = [_]c_int{ batch, heads, dim };
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, sliced, &out_shape, 3, s));
    return result;
}

/// Extract timestep t from a [B, T, H] tensor → [B, H]
fn sliceTimestep3(arr: mlx.mlx_array, batch: c_int, heads: c_int, t: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, t, 0 };
    const stop = [_]c_int{ batch, t + 1, heads };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, arr, &start, 3, &stop, 3, &strides, 3, s));
    const out_shape = [_]c_int{ batch, heads };
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, sliced, &out_shape, 2, s));
    return result;
}

fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

/// Build a "<container>.<leaf>" layer-weight suffix into `buf`. Used where the
/// stacked-MoE-expert container is named differently across converters — hy_v3
/// experts are `mlp.experts.*` in ox-ox-style MLX builds and `mlp.switch_mlp.*`
/// in mlx-lm builds (mlx-community Hy3-oQ2*, pipenetwork Hy3-REAP*), same
/// [E, out, in] tensor either way. Uses its own buffer so it can compose with
/// getLayerWeight's separate key buffer in one expression.
fn moeExpertSuffix(buf: []u8, container: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ container, leaf }) catch unreachable;
}

/// Resolve the hy_v3 stacked-expert container name for a layer. ox-ox-style MLX
/// conversions stack the experts under `mlp.experts.*`; mlx-lm's converter
/// (mlx-community Hy3-oQ2*, pipenetwork Hy3-REAP*) names the SAME [E, out, in]
/// tensors `mlp.switch_mlp.*`. Prefers `experts` when present, so an ox-ox
/// checkpoint binds byte-identically to before this fallback existed.
fn hy3ExpertContainer(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32) []const u8 {
    return if (getLayerWeightOpt(weights, buf, prefix, layer, "mlp.experts.gate_proj.weight") != null)
        "mlp.experts"
    else
        "mlp.switch_mlp";
}

/// Fetch a quantization scale/bias tensor, tolerant of dense bf16 models.
/// quant_bits == 0 ⇒ dense bf16 (no .scales/.biases exist anywhere) ⇒ return a
/// null-ctx array, which downstream code (qmatmulBits, gatherExpertMm, append
/// guards) reads as "this weight is plain bf16". quant_bits > 0 ⇒ a genuinely
/// quantized model, so fetch mandatorily — a missing scale stays a clear
/// MISSING WEIGHT error rather than silently degrading to a dense path.
fn getLayerScaleOrEmpty(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return getLayerWeight(weights, buf, prefix, layer, suffix);
}

/// Fetch a `.biases` tensor with mode-aware mandatoriness:
/// - dense bf16 (quant_bits == 0): null-ctx placeholder, like the scales.
/// - affine mode: mandatory — a missing bias is a clear MISSING WEIGHT error.
/// - bias-less modes (nvfp4/mxfp4/mxfp8, issue #24): OPTIONAL. The fp8
///   tensors ship no biases, but mixed QAT checkpoints override some layers
///   to affine (e.g. shared MLP at 8-bit/gs64) and those overrides DO carry
///   biases that the affine matmul needs.
fn getLayerBias(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, config: *const ModelConfig) mlx.mlx_array {
    if (config.quant_bits == 0) return mlx.mlx_array_new();
    if (config.quant_mode.hasBiases()) return getLayerWeight(weights, buf, prefix, layer, suffix);
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
}

/// Optional-typed variant for fields stored as `?mlx_array` (e.g. PLE projections).
/// Dense bf16 ⇒ `some(null-ctx)` so call sites that unwrap with `.?` still get a
/// valid (empty) array that qmatmul reads as bf16. Quantized ⇒ optional fetch.
fn getLayerScaleOrEmptyOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) ?mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix);
}

fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

fn getBertWeight(weights: *const Weights, buf: *[256]u8, name: []const u8) mlx.mlx_array {
    const n = std.fmt.bufPrint(buf, "{s}", .{name}) catch unreachable;
    return weights.get(n) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{n});
        unreachable;
    };
}

fn getBertLayerWeight(weights: *const Weights, buf: *[256]u8, layer: u32, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "encoder.layer.{d}.{s}", .{ layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

fn initBertLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8) ![]BertLayerWeights {
    log.info("Precomputing BERT layer weights...\n", .{});
    const layers = try allocator.alloc(BertLayerWeights, config.num_hidden_layers);

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

        lw.q_w = getBertLayerWeight(weights, name_buf, li, "attention.self.query.weight");
        lw.q_s = getBertLayerWeight(weights, name_buf, li, "attention.self.query.scales");
        lw.q_b = getBertLayerWeight(weights, name_buf, li, "attention.self.query.biases");
        lw.q_bias = getBertLayerWeight(weights, name_buf, li, "attention.self.query.bias");
        lw.k_w = getBertLayerWeight(weights, name_buf, li, "attention.self.key.weight");
        lw.k_s = getBertLayerWeight(weights, name_buf, li, "attention.self.key.scales");
        lw.k_b = getBertLayerWeight(weights, name_buf, li, "attention.self.key.biases");
        lw.k_bias = getBertLayerWeight(weights, name_buf, li, "attention.self.key.bias");
        lw.v_w = getBertLayerWeight(weights, name_buf, li, "attention.self.value.weight");
        lw.v_s = getBertLayerWeight(weights, name_buf, li, "attention.self.value.scales");
        lw.v_b = getBertLayerWeight(weights, name_buf, li, "attention.self.value.biases");
        lw.v_bias = getBertLayerWeight(weights, name_buf, li, "attention.self.value.bias");
        lw.o_w = getBertLayerWeight(weights, name_buf, li, "attention.output.dense.weight");
        lw.o_s = getBertLayerWeight(weights, name_buf, li, "attention.output.dense.scales");
        lw.o_b = getBertLayerWeight(weights, name_buf, li, "attention.output.dense.biases");
        lw.o_bias = getBertLayerWeight(weights, name_buf, li, "attention.output.dense.bias");
        lw.attn_norm_w = getBertLayerWeight(weights, name_buf, li, "attention.output.LayerNorm.weight");
        lw.attn_norm_b = getBertLayerWeight(weights, name_buf, li, "attention.output.LayerNorm.bias");
        lw.inter_w = getBertLayerWeight(weights, name_buf, li, "intermediate.dense.weight");
        lw.inter_s = getBertLayerWeight(weights, name_buf, li, "intermediate.dense.scales");
        lw.inter_b = getBertLayerWeight(weights, name_buf, li, "intermediate.dense.biases");
        lw.inter_bias = getBertLayerWeight(weights, name_buf, li, "intermediate.dense.bias");
        lw.out_w = getBertLayerWeight(weights, name_buf, li, "output.dense.weight");
        lw.out_s = getBertLayerWeight(weights, name_buf, li, "output.dense.scales");
        lw.out_b = getBertLayerWeight(weights, name_buf, li, "output.dense.biases");
        lw.out_bias = getBertLayerWeight(weights, name_buf, li, "output.dense.bias");
        lw.out_norm_w = getBertLayerWeight(weights, name_buf, li, "output.LayerNorm.weight");
        lw.out_norm_b = getBertLayerWeight(weights, name_buf, li, "output.LayerNorm.bias");
    }
    return layers;
}

fn initBert(io: std.Io, allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !Transformer {
    // Word embeddings (reuse standard emb_w/s/b fields)
    const emb_w = getBertWeight(weights, name_buf, "embeddings.word_embeddings.weight");
    const emb_s = getBertWeight(weights, name_buf, "embeddings.word_embeddings.scales");
    const emb_b = getBertWeight(weights, name_buf, "embeddings.word_embeddings.biases");

    // Position embeddings
    const pos_w = getBertWeight(weights, name_buf, "embeddings.position_embeddings.weight");
    const pos_s = getBertWeight(weights, name_buf, "embeddings.position_embeddings.scales");
    const pos_b = getBertWeight(weights, name_buf, "embeddings.position_embeddings.biases");

    // Token type embeddings
    const toktype_w = getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.weight");
    const toktype_s = getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.scales");
    const toktype_b = getBertWeight(weights, name_buf, "embeddings.token_type_embeddings.biases");

    // Embedding LayerNorm
    const emb_norm_w = getBertWeight(weights, name_buf, "embeddings.LayerNorm.weight");
    const emb_norm_b = getBertWeight(weights, name_buf, "embeddings.LayerNorm.bias");

    const bert_layers = try initBertLayers(allocator, config, weights, name_buf);

    // Batch eval all BERT weights
    {
        const eval_start = std.Io.Timestamp.now(io, .awake);
        const all_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all_vec);

        _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_s);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_b);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_s);
        _ = mlx.mlx_vector_array_append_value(all_vec, pos_b);
        _ = mlx.mlx_vector_array_append_value(all_vec, toktype_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_norm_w);
        _ = mlx.mlx_vector_array_append_value(all_vec, emb_norm_b);

        for (bert_layers) |lw| {
            inline for (comptime structFields(BertLayerWeights)) |field| {
                _ = mlx.mlx_vector_array_append_value(all_vec, @field(lw, field.name));
            }
        }

        try mlx.check(mlx.mlx_eval(all_vec));
        const eval_ms: i64 = @intCast(@divTrunc(eval_start.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
        log.info("Batch eval all weights: {d}ms\n", .{eval_ms});
    }

    const cache = try KVCache.init(allocator, 0);

    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = emb_w,
        .emb_s = emb_s,
        .emb_b = emb_b,
        .emb_scale = null,
        .final_norm = mlx.mlx_array_new(),
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .layers = &.{},
        .owns_lm_head = false,
        .owns_norms = false,
        .gelu_coeff = bf16Scalar(0.7978845608028654, s),
        .gelu_inner = bf16Scalar(0.044715, s),
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .three = bf16Scalar(3.0, s),
        .neg_one = null,
        .ple_emb_w = mlx.mlx_array_new(),
        .ple_emb_s = mlx.mlx_array_new(),
        .ple_emb_b = mlx.mlx_array_new(),
        .ple_proj_w = mlx.mlx_array_new(),
        .ple_proj_s = mlx.mlx_array_new(),
        .ple_proj_b = mlx.mlx_array_new(),
        .ple_proj_norm = mlx.mlx_array_new(),
        .ple_proj_quantized = false,
        .softcap_scalar = null,
        .v_norm_weight = null,
        .v_norm_weight_global = null,
        .rope_freqs_global = null,
        .bert_layers = bert_layers,
        .bert_pos_w = pos_w,
        .bert_pos_s = pos_s,
        .bert_pos_b = pos_b,
        .bert_toktype_w = toktype_w,
        .bert_toktype_s = toktype_s,
        .bert_toktype_b = toktype_b,
        .bert_emb_norm_w = emb_norm_w,
        .bert_emb_norm_b = emb_norm_b,
        .moe_layers = null,
        .ssm_entries = null,
        .moe_seq_offset = 0,
        .hybrid_layers = null,
        .embedding_norm = null,
        .prompt_cache = null,
    };
}

fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}

// ── Tests ──

const testing = std.testing;

/// Helper: create a dummy K or V tensor of shape [1, 1, seq_len, 1].
fn testKV(seq_len: usize, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 1, sl, 1 };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

test "KVCache sliding window views return last max_seq entries" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Simulate 3 prefill tokens (max_seq=4 sliding window)
    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    // After prefill: offset=3, step=3
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);

    // Decode token 4 — still within window
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.step);

    // Decode token 5 — exceeds window, but buffer grows (no trimming)
    // Views return last 4 entries only
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        // View should be 4 entries (max_seq), not 5
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    // Buffer has 5 entries, but view shows 4. step=5 (absolute).
    try testing.expectEqual(@as(usize, 5), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Decode tokens 6,7,8 — step keeps incrementing, views stay at max_seq
    for (0..3) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 8), cache.step);
}

test "KVCache step resets on truncate" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add 5 tokens
    {
        const k = testKV(5, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(5, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Truncate to 3 (simulating KV cache reuse)
    try cache.truncate(3, s);
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);
}

test "KVCache step without trimming matches offset" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add tokens without max_seq (no trimming)
    for (0..10) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    // Without trimming, step and offset should be equal
    try testing.expectEqual(@as(usize, 10), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 10), cache.step);
}

test "KVCache step with multi-layer only increments once per update" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();

    // Update all 3 layers with 2 tokens each
    for (0..3) |layer| {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    // step should be 2 (one sequence worth), not 6 (3 layers × 2)
    try testing.expectEqual(@as(usize, 2), cache.step);
}

// ── Cache snapshot/restore (for spec-decode rollback) ──────────────────────

test "KVCache snapshot/restore round-trip preserves entries and step" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    // Build state: 4 tokens across 2 layers.
    for (0..2) |layer| {
        const k = testKV(4, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(4, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);

    // Snapshot at this point.
    var snap = try cache.snapshot();
    defer snap.deinit();

    // Mutate cache: add 2 more tokens to layer 0 only (to verify per-layer
    // offset is captured, not just global step).
    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 6), cache.step);
    try testing.expectEqual(@as(usize, 6), cache.entries[0].offset);

    // Restore — step and per-layer offsets revert to snapshot.
    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.entries[1].offset);
}

test "KVCache snapshot then more updates does not corrupt the snapshot" {
    // Critical invariant for spec-decode rollback: if we snapshot, then verify, then
    // (rejected) restore, snapshot must NOT have been mutated by the intervening
    // updates. The buffer is shared via refcount but must not be aliased through
    // the cache entry pointer.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }

    var snap = try cache.snapshot();
    defer snap.deinit();
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);

    // Run several updates to grow the cache buffer (forces buffer reallocation
    // inside update(), which would invalidate a naive snapshot).
    for (0..6) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);

    // Snapshot still reports its captured state.
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), snap.step);

    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 2), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), cache.step);
}

test "KVCache snapshot/restore in a tight loop does not leak" {
    // testing.allocator is a TrackingAllocator — any unfreed allocation here
    // surfaces as a test failure at the leak-detection step.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        const k2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(k2);
        const v2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(v2);
        _ = try cache.update(1, k2, v2, s, 0);
    }

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var snap = try cache.snapshot();
        defer snap.deinit();
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        try cache.restore(&snap);
    }
}

test "SSMCacheEntry snapshot/restore round-trip preserves arrays" {
    const s = mlx.gpuStream();
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    // Populate with arbitrary state.
    const conv_shape = [_]c_int{ 1, 3, 4 };
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &conv_shape, 3, .float32, s));

    const ssm_shape = [_]c_int{ 1, 2, 8, 4 };
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &ssm_shape, 4, .float32, s));
    entry.initialized = true;

    var snap = ssmSnapshot(&entry);
    defer ssmSnapshotDeinit(&snap);

    // Mutate: replace ssm_state with a different shape.
    _ = mlx.mlx_array_free(entry.ssm_state);
    entry.ssm_state = mlx.mlx_array_new();
    const new_shape = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_zeros(&entry.ssm_state, &new_shape, 4, .float32, s));

    try ssmRestore(&entry, &snap);
    try testing.expect(entry.initialized);
    const restored_shape = mlx.getShape(entry.ssm_state);
    try testing.expectEqual(@as(c_int, 1), restored_shape[0]);
    try testing.expectEqual(@as(c_int, 2), restored_shape[1]);
    try testing.expectEqual(@as(c_int, 8), restored_shape[2]);
    try testing.expectEqual(@as(c_int, 4), restored_shape[3]);
}

test "SSMCacheEntry snapshot/restore handles null ssm_state (LFM2 gated_conv)" {
    // LFM2 `gatedConv` populates `conv_state` but never `ssm_state` — the
    // gated-convolution layer doesn't have a recurrence state, only a
    // convolution-window cache. The snapshot/restore code must NOT crash on
    // this shape (`initialized=true`, `conv_state` non-null, `ssm_state.ctx`
    // null). This was the root cause of the Workstream D PLD-on-hybrid bug.
    const s = mlx.gpuStream();
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(), // stays null — no LFM2 layer ever touches it
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    // Simulate `conv1dWithCache` having run once: conv_state populated,
    // initialized=true, ssm_state still null (its ctx is null).
    const conv_shape = [_]c_int{ 1, 3, 4 };
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &conv_shape, 3, .float32, s));
    entry.initialized = true;
    try testing.expect(entry.ssm_state.ctx == null);

    // Snapshot must succeed without dereferencing the null ssm_state.
    var snap = ssmSnapshot(&entry);
    defer ssmSnapshotDeinit(&snap);
    try testing.expect(snap.initialized);
    try testing.expect(snap.conv_state.ctx != null);
    try testing.expect(snap.ssm_state.ctx == null);

    // Mutate conv_state in `entry`, then restore — restore must rebind
    // conv_state without crashing on the still-null ssm_state.
    _ = mlx.mlx_array_free(entry.conv_state);
    entry.conv_state = mlx.mlx_array_new();
    const mutated_shape = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &mutated_shape, 3, .float32, s));

    try ssmRestore(&entry, &snap);
    try testing.expect(entry.initialized);
    try testing.expect(entry.ssm_state.ctx == null); // still null after restore
    const restored = mlx.getShape(entry.conv_state);
    try testing.expectEqual(@as(c_int, 1), restored[0]);
    try testing.expectEqual(@as(c_int, 3), restored[1]);
    try testing.expectEqual(@as(c_int, 4), restored[2]);
}

test "captureSsmCheckpoint materializes state copies (parent-buffer retention class)" {
    // Hot-cache SSM checkpoints used to refcount-share the live conv/ssm
    // state handles. Those are routinely SLICES of much larger parent buffers
    // (e.g. the prefill chunk's conv input [B,(k-1)+T,C]), so a committed
    // entry silently retained the whole parent — the measured ~3.4x
    // "[hot-cache] resident" under-count on hybrid archs. Capture must
    // materialize an OWNED copy: the checkpoint's data pointer may not alias
    // the parent's buffer, and ssmCheckpointBytes must equal true retention.
    const s = mlx.gpuStream();

    // Parent buffer: [1, 1024, 8] f32, 32 KB.
    var parent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(parent);
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, 8192.0, 1.0, .float32, s));
        const shape = [_]c_int{ 1, 1024, 8 };
        try mlx.check(mlx.mlx_reshape(&parent, flat, &shape, 3, s));
        _ = mlx.mlx_array_eval(parent);
    }

    // conv_state := parent[:, 1021:1024, :] — after eval this is a
    // shared-buffer view into `parent` (MLX slice keeps the parent alive).
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = true,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }
    {
        const st = [_]c_int{ 0, 1021, 0 };
        const sp = [_]c_int{ 1, 1024, 8 };
        const sd = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&entry.conv_state, parent, &st, 3, &sp, 3, &sd, 3, s));
        _ = mlx.mlx_array_eval(entry.conv_state);
    }

    var entries = [_]SSMCacheEntry{entry};
    var cp = try captureSsmCheckpoint(testing.allocator, &entries, 3, s);
    defer cp.deinit(testing.allocator);

    // The checkpoint's conv_state must live in its own buffer, outside the
    // parent's data range.
    _ = mlx.mlx_array_eval(cp.layers[0].conv_state);
    const parent_ptr = mlx.mlx_array_data_float32(parent).?;
    const parent_bytes = @as(u64, mlx.mlx_array_size(parent)) * @as(u64, mlx.mlx_array_itemsize(parent));
    const cp_ptr = mlx.mlx_array_data_float32(cp.layers[0].conv_state).?;
    const p0 = @intFromPtr(parent_ptr);
    const p1 = p0 + parent_bytes;
    const c0 = @intFromPtr(cp_ptr);
    try testing.expect(c0 < p0 or c0 >= p1);

    // Values must survive the copy: parent row 1021 starts at 1021*8.
    try testing.expectEqual(@as(f32, 1021.0 * 8.0), cp_ptr[0]);
    try testing.expectEqual(@as(f32, 1021.0 * 8.0 + 23.0), cp_ptr[23]);

    // Accounting matches true retention: 3*8 f32 elements, nothing more.
    try testing.expectEqual(@as(u64, 3 * 8 * 4), ssmCheckpointBytes(&cp));

    // Null ssm_state stays null (LFM2 gated_conv shape).
    try testing.expect(cp.layers[0].ssm_state.ctx == null);
}

test "affineParamsFromGeometry: exact per-weight solve for off-config sidecar quants" {
    // Real geometries from the stamsam 35B-A3B MTP sidecar over a 4-bit/gs-64
    // affine trunk: q_proj 5-bit/gs-128, v_proj 6-bit/gs-128, shared expert
    // 8-bit/gs-128, switch experts 4-bit/gs-64. The old config-gs detection
    // (detectQuantBits with gs 64) returns garbage for the gs-128 tensors.
    const s = mlx.gpuStream();
    const H = 2048; // sidecar hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // q_proj: w [8192, 320] u32, scales [8192, 16] bf16 → 5-bit / gs 128.
    const wq = try mk.arr(&.{ 8192, 320 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wq);
    const sq = try mk.arr(&.{ 8192, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sq);
    const qp_q = affineParamsFromGeometry(wq, sq, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 5), qp_q.bits);
    try testing.expectEqual(@as(u32, 128), qp_q.group_size);

    // v_proj: w [512, 384] u32, scales [512, 16] bf16 → 6-bit / gs 128.
    const wv = try mk.arr(&.{ 512, 384 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wv);
    const sv = try mk.arr(&.{ 512, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sv);
    const qp_v = affineParamsFromGeometry(wv, sv, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 6), qp_v.bits);
    try testing.expectEqual(@as(u32, 128), qp_v.group_size);

    // Switch experts (3D): w [256, 512, 256] u32, scales [256, 512, 32] bf16
    // → 4-bit / gs 64 (trunk-standard — must still solve).
    const we = try mk.arr(&.{ 256, 512, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 256, 512, 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = affineParamsFromGeometry(we, se, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 4), qp_e.bits);
    try testing.expectEqual(@as(u32, 64), qp_e.group_size);

    // Degenerate: bits wouldn't divide exactly → null (caller falls back).
    const qp_bad = affineParamsFromGeometry(wq, sq, 3000);
    try testing.expect(qp_bad == null);

    // And through computeQuantParams on an AFFINE gs-64 config: the hinted
    // exact solve must WIN over the config group size (pre-fix this returned
    // bits 10 / gs 64 for the q_proj geometry).
    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qp_via = computeQuantParams(&cfg, wq, sq, H);
    try testing.expectEqual(@as(u32, 5), qp_via.bits);
    try testing.expectEqual(@as(u32, 128), qp_via.group_size);
}

test "computeQuantParams resolves mixed nvfp4 + affine-override weights" {
    // Real-world shape: gemma-4 QAT nvfp4 checkpoints quantize most tensors
    // nvfp4 (uint8 fp8 scales, gs 16) but override the shared MLP to affine
    // 8-bit/gs64 (bf16 scales + biases). Detection key: scales dtype, then
    // the activation inner dim to pin (bits, group_size) for the override.
    const s = mlx.gpuStream();
    const IN = 2560;
    const OUT = 8;

    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 16;
    cfg.quant_mode = .nvfp4;

    // nvfp4 tensor: w [OUT, IN*4/32] u32, scales [OUT, IN/16] u8.
    var w_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_nv);
    var sc_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_nv);
    const w_nv_sh = [_]c_int{ OUT, IN * 4 / 32 };
    const sc_nv_sh = [_]c_int{ OUT, IN / 16 };
    try mlx.check(mlx.mlx_zeros(&w_nv, &w_nv_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_nv, &sc_nv_sh, 2, .uint8, s));
    const qp_nv = computeQuantParams(&cfg, w_nv, sc_nv, null);
    try testing.expectEqual(QuantMode.nvfp4, qp_nv.mode);
    try testing.expectEqual(@as(u32, 4), qp_nv.bits);
    try testing.expectEqual(@as(u32, 16), qp_nv.group_size);

    // Affine 8-bit/gs64 override: w [OUT, IN*8/32] u32, scales [OUT, IN/64] bf16.
    var w_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_af);
    var sc_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_af);
    const w_af_sh = [_]c_int{ OUT, IN * 8 / 32 };
    const sc_af_sh = [_]c_int{ OUT, IN / 64 };
    try mlx.check(mlx.mlx_zeros(&w_af, &w_af_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_af, &sc_af_sh, 2, .bfloat16, s));

    // With the activation inner-dim hint: exact.
    const qp_af = computeQuantParams(&cfg, w_af, sc_af, IN);
    try testing.expectEqual(QuantMode.affine, qp_af.mode);
    try testing.expectEqual(@as(u32, 8), qp_af.bits);
    try testing.expectEqual(@as(u32, 64), qp_af.group_size);

    // Without a hint: falls back to mlx-lm's override default gs=64.
    const qp_af_nohint = computeQuantParams(&cfg, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_af_nohint.mode);
    try testing.expectEqual(@as(u32, 8), qp_af_nohint.bits);
    try testing.expectEqual(@as(u32, 64), qp_af_nohint.group_size);

    // Plain affine model (config mode affine, bf16 scales): unchanged
    // detect-against-config-gs behavior.
    var cfg_affine = ModelConfig{};
    cfg_affine.quant_bits = 4;
    cfg_affine.quant_group_size = 64;
    cfg_affine.quant_mode = .affine;
    const qp_plain = computeQuantParams(&cfg_affine, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_plain.mode);
    try testing.expectEqual(@as(u32, 8), qp_plain.bits);
    try testing.expectEqual(@as(u32, 64), qp_plain.group_size);
}

test "QuantParamsCache put/lookup round-trip" {
    var cache: BitsCache = .{};
    const s = mlx.gpuStream();

    // Three real arrays with distinct ctx pointers — the cache keys.
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    const shape = [_]c_int{8};
    try mlx.check(mlx.mlx_zeros(&a, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&b, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&c, &shape, 1, .bfloat16, s));

    try testing.expect(cache.put(a.ctx.?, .{ .bits = 4, .group_size = 32, .mode = .affine }));
    try testing.expect(cache.put(b.ctx.?, .{ .bits = 4, .group_size = 64, .mode = .nvfp4 }));
    try testing.expect(cache.put(c.ctx.?, .{ .bits = 8, .group_size = 128, .mode = .affine }));

    // Build a Transformer-like view over the cache via quantParamsFor — we
    // need a real Transformer to call the method, so verify by direct slot
    // lookup instead. (The forward path goes through quantParamsFor, which is
    // exercised by integration tests; this test pins the data structure.)
    {
        const idx = BitsCache.slot(a.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == a.ctx) {
                try testing.expectEqual(@as(u8, 4), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 32 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@intFromEnum(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
    {
        const idx = BitsCache.slot(c.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == c.ctx) {
                try testing.expectEqual(@as(u8, 8), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 128 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@intFromEnum(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "qmatmulBits dispatches to plain matmul when scales has null ctx (Unsloth Dynamic bf16)" {
    const s = mlx.gpuStream();

    // x: [1, 1, in=4], w: [out=2, in=4] (PyTorch convention).
    // Expected: x @ w.T = [1, 1, 2]
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_arange(&x_flat, 0.0, 4.0, 1.0, .float32, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    {
        const sh = [_]c_int{ 1, 1, 4 };
        try mlx.check(mlx.mlx_reshape(&x, x_flat, &sh, 3, s));
    }

    var w_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_flat);
    try mlx.check(mlx.mlx_arange(&w_flat, 0.0, 8.0, 1.0, .float32, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    {
        const sh = [_]c_int{ 2, 4 };
        try mlx.check(mlx.mlx_reshape(&w, w_flat, &sh, 2, s));
    }

    // Pre-transpose like initMoeLayers does for null-scales weights: [out, in] → [in, out]
    const w_t = try transposeBf16Weight(w, s);
    defer _ = mlx.mlx_array_free(w_t);

    // qmatmulBits with null sc/bi must plain-matmul x @ w_t.
    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const got = try qmatmulBits(x, w_t, null_sc, null_bi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(got);

    // Reduce to host floats for comparison.
    var got_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_f32);
    try mlx.check(mlx.mlx_astype(&got_f32, got, .float32, s));
    var got_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_flat);
    {
        const sh = [_]c_int{2};
        try mlx.check(mlx.mlx_reshape(&got_flat, got_f32, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, got_flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const data = mlx.mlx_array_data_float32(got_flat) orelse return error.TestUnexpectedNullData;
    // [0,1,2,3] @ [[0,4],[1,5],[2,6],[3,7]] = [0+1*1+2*2+3*3, 0+1*5+2*6+3*7] = [14, 38]
    // (w transposed from [[0,1,2,3],[4,5,6,7]] gives w_t = [[0,4],[1,5],[2,6],[3,7]])
    try testing.expectApproxEqAbs(@as(f32, 14.0), data[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 38.0), data[1], 1e-3);
}

/// Test helper: eval `arr`, flatten, and copy the first `out.len` f32 values to host.
fn testReadF32(arr: mlx.mlx_array, out: []f32, s: mlx.mlx_stream) !void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const sh = [_]c_int{@intCast(out.len)};
    try mlx.check(mlx.mlx_reshape(&flat, f, &sh, 1, s));
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    _ = mlx.mlx_vector_array_append_value(ev, flat);
    try mlx.check(mlx.mlx_eval(ev));
    const data = mlx.mlx_array_data_float32(flat) orelse return error.TestUnexpectedNullData;
    @memcpy(out, data[0..out.len]);
}

test "hy3ExpertContainer resolves both mlx-lm switch_mlp and ox-ox experts naming" {
    // hy_v3 MoE experts stack under `mlp.experts.*` in ox-ox-style MLX builds
    // but `mlp.switch_mlp.*` in mlx-lm builds (mlx-community Hy3-oQ2e,
    // pipenetwork Hy3-REAP*). Same [E, out, in] tensor; the loader must accept
    // either name or it crashes with MISSING WEIGHT at load (live 2026-07-16).
    const allocator = testing.allocator;
    var buf: [256]u8 = undefined;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // mlx-lm build: only switch_mlp present → resolver picks it (was the crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.switch_mlp.gate_proj.weight");
        try testing.expectEqualStrings("mlp.switch_mlp", hy3ExpertContainer(&w, &buf, "model", 1));
    }
    // ox-ox build: experts present → preferred, so it binds byte-identically.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.layers.1.mlp.experts.gate_proj.weight");
        try testing.expectEqualStrings("mlp.experts", hy3ExpertContainer(&w, &buf, "model", 1));
    }
    // The suffix builder composes the resolved container with each leaf.
    var sbuf: [64]u8 = undefined;
    try testing.expectEqualStrings(
        "mlp.switch_mlp.down_proj.scales",
        moeExpertSuffix(&sbuf, "mlp.switch_mlp", "down_proj.scales"),
    );
}

test "splitPackedGateUp slices DiffusionGemma experts.gate_up_proj into gate/up halves" {
    // DiffusionGemma packs each expert's gate and up projections in one
    // tensor [E, 2*M, X] with gate rows first (HF chunks the gate_up output
    // in halves: gate = [..., :M], up = [..., M:]). The split must slice
    // axis 1 so the same code serves the packed quantized weight, its
    // scales, and its biases (all share the [E, rows, X] layout).
    const s = mlx.gpuStream();
    const E = 2;
    const M = 2; // moe_intermediate rows per half
    const X = 3;

    var host: [E * 2 * M * X]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @floatFromInt(i);
    const sh = [_]c_int{ E, 2 * M, X };
    const packed_arr = mlx.mlx_array_new_data(&host, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(packed_arr);

    const pair = try splitPackedGateUp(packed_arr, s);
    defer _ = mlx.mlx_array_free(pair.gate);
    defer _ = mlx.mlx_array_free(pair.up);

    const gshape = mlx.getShape(pair.gate);
    const ushape = mlx.getShape(pair.up);
    try testing.expectEqualSlices(c_int, &[_]c_int{ E, M, X }, gshape);
    try testing.expectEqualSlices(c_int, &[_]c_int{ E, M, X }, ushape);

    // Expert e's gate = rows [0, M) of its 2M block; up = rows [M, 2M).
    var gate_host: [E * M * X]f32 = undefined;
    var up_host: [E * M * X]f32 = undefined;
    try testReadF32(pair.gate, &gate_host, s);
    try testReadF32(pair.up, &up_host, s);
    for (0..E) |e| {
        for (0..M) |r| {
            for (0..X) |c| {
                const out_idx = (e * M + r) * X + c;
                const gate_src: f32 = @floatFromInt((e * 2 * M + r) * X + c);
                const up_src: f32 = @floatFromInt((e * 2 * M + M + r) * X + c);
                try testing.expectApproxEqAbs(gate_src, gate_host[out_idx], 0.001);
                try testing.expectApproxEqAbs(up_src, up_host[out_idx], 0.001);
            }
        }
    }
}

test "splitPackedGateUp halves feed gather_qmm correctly (lazy-slice zeros regression)" {
    // CLASS-BUG regression: mlx_slice produces a lazy VIEW with parent
    // strides; feeding such a view to mlx_gather_qmm silently produced
    // all-ZERO expert outputs (gather_qmm assumes row-contiguous weight
    // buffers). Reading the slice via data pointers materializes it
    // correctly, so a value-equality test on the split alone cannot catch
    // this — the assertion must go THROUGH gather_qmm. splitPackedGateUp
    // now materializes contiguous halves; this test pins that by comparing
    // gather_qmm on the split gate half against gather_qmm on an
    // independently-built contiguous copy of the same rows.
    const s = mlx.gpuStream();
    // Geometry matters: small/toy shapes take a strided-tolerant kernel path
    // and do NOT reproduce the zeros — keep dims near the real checkpoint's
    // 4-bit/gs64 layout (full scale is E=128, M=704, IN=2816).
    const E = 16;
    const M = 128; // rows per half
    const IN = 512;
    const gs = 64;
    const bits = 4;

    // Dense [E, 2M, IN] bf16 source, then quantize.
    const w_host = try std.testing.allocator.alloc(f32, E * 2 * M * IN);
    defer std.testing.allocator.free(w_host);
    for (w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8))) * 0.07;
    const wsh = [_]c_int{ E, 2 * M, IN };
    const w_f32 = mlx.mlx_array_new_data(w_host.ptr, &wsh, 3, .float32);
    defer _ = mlx.mlx_array_free(w_f32);
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&q, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));

    const q_pair = try splitPackedGateUp(q, s);
    defer _ = mlx.mlx_array_free(q_pair.gate);
    defer _ = mlx.mlx_array_free(q_pair.up);
    const s_pair = try splitPackedGateUp(sc, s);
    defer _ = mlx.mlx_array_free(s_pair.gate);
    defer _ = mlx.mlx_array_free(s_pair.up);
    const b_pair = try splitPackedGateUp(bi, s);
    defer _ = mlx.mlx_array_free(b_pair.gate);
    defer _ = mlx.mlx_array_free(b_pair.up);

    // Independent contiguous ground truth: quantize the gate rows directly.
    const g_host = try std.testing.allocator.alloc(f32, E * M * IN);
    defer std.testing.allocator.free(g_host);
    for (0..E) |e| {
        for (0..M) |r| {
            for (0..IN) |c| {
                g_host[(e * M + r) * IN + c] = w_host[(e * 2 * M + r) * IN + c];
            }
        }
    }
    const gsh = [_]c_int{ E, M, IN };
    const g_f32 = mlx.mlx_array_new_data(g_host.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g_f32);
    var g_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_bf16);
    try mlx.check(mlx.mlx_astype(&g_bf16, g_f32, .bfloat16, s));
    var gvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(gvec);
    try mlx.check(mlx.mlx_quantize(&gvec, g_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var gq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gq);
    var gsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gsc);
    var gbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gbi);
    try mlx.check(mlx.mlx_vector_array_get(&gq, gvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&gsc, gvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&gbi, gvec, 2));

    // gather_qmm through both, on BOTH dispatch shapes moeMLP2 uses:
    //   decode:  x [1,1,1,1,IN], inds [1,1,1], sorted=false
    //   prefill: x [N,1,IN], sorted rhs inds [N], sorted=true  ← the shape
    //            the live zeros bug fired on
    var x_host: [IN]f32 = undefined;
    for (&x_host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.11 - 0.3;

    // decode shape
    {
        const xsh = [_]c_int{ 1, 1, 1, 1, IN };
        const x_f32 = mlx.mlx_array_new_data(&x_host, &xsh, 5, .float32);
        defer _ = mlx.mlx_array_free(x_f32);
        var x_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf16);
        try mlx.check(mlx.mlx_astype(&x_bf16, x_f32, .bfloat16, s));

        for (0..E) |e| {
            var inds_host = [_]u32{@intCast(e)};
            const ish = [_]c_int{ 1, 1, 1 };
            const inds = mlx.mlx_array_new_data(&inds_host, &ish, 3, .uint32);
            defer _ = mlx.mlx_array_free(inds);
            const no_idx = mlx.mlx_array{ .ctx = null };

            var out_split = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(out_split);
            try gatherExpertMm(&out_split, x_bf16, q_pair.gate, s_pair.gate, b_pair.gate, no_idx, inds, bits, gs, .affine, false, s);
            var out_ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(out_ref);
            try gatherExpertMm(&out_ref, x_bf16, gq, gsc, gbi, no_idx, inds, bits, gs, .affine, false, s);

            var split_host: [M]f32 = undefined;
            var ref_host: [M]f32 = undefined;
            try testReadF32(out_split, &split_host, s);
            try testReadF32(out_ref, &ref_host, s);
            var nonzero = false;
            for (split_host, ref_host) |a, b| {
                try testing.expectApproxEqAbs(b, a, 0.02);
                if (@abs(a) > 0.001) nonzero = true;
            }
            // The original bug produced exact zeros — make that explicit.
            try testing.expect(nonzero);
        }
    }

    // sorted prefill shape: N=E rows, one per expert, sorted inds [0..E)
    {
        const xr_host = try std.testing.allocator.alloc(f32, E * IN);
        defer std.testing.allocator.free(xr_host);
        for (xr_host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.2;
        const xsh = [_]c_int{ E, 1, IN };
        const x_f32 = mlx.mlx_array_new_data(xr_host.ptr, &xsh, 3, .float32);
        defer _ = mlx.mlx_array_free(x_f32);
        var x_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf16);
        try mlx.check(mlx.mlx_astype(&x_bf16, x_f32, .bfloat16, s));

        var inds_host: [E]u32 = undefined;
        for (&inds_host, 0..) |*v, i| v.* = @intCast(i);
        const ish = [_]c_int{E};
        const inds = mlx.mlx_array_new_data(&inds_host, &ish, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const no_idx = mlx.mlx_array{ .ctx = null };

        var out_split = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_split);
        try gatherExpertMm(&out_split, x_bf16, q_pair.gate, s_pair.gate, b_pair.gate, no_idx, inds, bits, gs, .affine, true, s);
        var out_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_ref);
        try gatherExpertMm(&out_ref, x_bf16, gq, gsc, gbi, no_idx, inds, bits, gs, .affine, true, s);

        const split_host = try std.testing.allocator.alloc(f32, E * M);
        defer std.testing.allocator.free(split_host);
        const ref_host = try std.testing.allocator.alloc(f32, E * M);
        defer std.testing.allocator.free(ref_host);
        try testReadF32(out_split, split_host, s);
        try testReadF32(out_ref, ref_host, s);
        var nonzero = false;
        for (split_host, ref_host) |a, b| {
            try testing.expectApproxEqAbs(b, a, 0.02);
            if (@abs(a) > 0.001) nonzero = true;
        }
        try testing.expect(nonzero);
    }
}

test "gatherExpertMm dense bf16 matches per-expert ground truth (decode + prefill shapes)" {
    // Centerpiece TDD gate for fully-dense bf16 MoE. Proves the dense gather_mm
    // path (gatherExpertMm with null scales + generalized transposeBf16Weight)
    // computes the same per-expert matmul as (a) an independent fp32 ground truth
    // and (b) the established quantized gather_qmm path — for BOTH the decode
    // (S=1, unsorted, 5D x) and prefill ([N,1,in], sorted) call shapes used by
    // moeMLP2. If the historical Qwen3.6-A3B-bf16 generation bug lived in the
    // expert gather, this test fails for the right reason.
    const s = mlx.gpuStream();
    const alloc = testing.allocator;

    const E = 4;
    const IN = 32; // must be a multiple of gs
    const OUT = 8;
    const gs = 32; // mlx_quantize supports group sizes 32, 64, 128
    const bits = 8; // near-lossless quant for a tight cross-check

    // Build w_orig [E, OUT, IN] bf16 from a deterministic small-valued buffer.
    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize → [q, sc, bi]; dequantize back so the dense path consumes the
    // exact numbers gather_qmm sees internally (cancels quant error in the
    // dense-vs-quant comparison).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    var q_bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_bi);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q_bi, qvec, 2));

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, q_bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    // w_deq_t [E, IN, OUT] — the dense weight layout the loader produces.
    const w_deq_t = try transposeBf16Weight(w_deq, s);
    defer _ = mlx.mlx_array_free(w_deq_t);

    // Read dequantized weights to host for ground truth: w_deq_host[e*OUT*IN + o*IN + k].
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const no_idx = mlx.mlx_array{ .ctx = null };

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        // Ground truth: gt[k*OUT + o] = sum_in x[in] * w_deq[inds[k]][o][in].
        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        // Dense path.
        var dense5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense5);
        try gatherExpertMm(&dense5, x_exp, w_deq_t, null_sc, null_bi, no_idx, inds, bits, gs, .affine, false, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense5, s)); // [K, OUT]
        const dense_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        // Quantized path (cross-check).
        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, q_bi, no_idx, inds, bits, gs, .affine, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s));
        const quant_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..K * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2); // dense == ground truth
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2); // quant agrees too
        }
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 }; // sorted experts
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        // Ground truth: gt[i*OUT + o] = sum_in x[i][in] * w_deq[sorted[i]][o][in].
        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var dense3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense3);
        try gatherExpertMm(&dense3, x_rep, w_deq_t, null_sc, null_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense3, s)); // [N, OUT]
        const dense_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, q_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        const quant_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..N * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2);
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2);
        }
    }
}

test "qmatmulBits: prefill-width affine calls take the dequant+GEMM route (engaged + no worse than stock)" {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xD0DE);
    const rnd = prng.random();
    const K: c_int = 256;
    const N: c_int = 384;
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);

    // Random bf16 weight quantized to affine q4 gs64 (the 27B trunk class).
    const wn: usize = @intCast(N * K);
    const wbuf = try al.alloc(f32, wn);
    for (wbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const wshape = [_]c_int{ N, K };
    const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
    al.free(wbuf);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    const xn: usize = @intCast(M * K);
    const xbuf = try al.alloc(f32, xn);
    for (xbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const xshape = [_]c_int{ 1, M, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
    al.free(xbuf);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    // f32 ground truth: fp32 dequant + fp32 GEMM.
    var wdq32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32);
    try mlx.check(mlx.mlx_dequantize(&wdq32, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
    var wdq32_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32_t);
    try mlx.check(mlx.mlx_transpose(&wdq32_t, wdq32, s));
    var gt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gt);
    try mlx.check(mlx.mlx_matmul(&gt, x32, wdq32_t, s));
    try mlx.check(mlx.mlx_array_eval(gt));
    const gtd = mlx.mlx_array_data_float32(gt) orelse return error.InvalidDtype;
    const on: usize = @intCast(M * N);

    // Stock qmm (route forced off).
    prefill_dq_gemm_override = false;
    const stock = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(stock);
    const stock_f = try evalToF32(al, stock, on, s);
    defer al.free(stock_f);

    // Routed (default on): engagement must be COUNTED.
    prefill_dq_gemm_override = true;
    defer prefill_dq_gemm_override = null;
    const before = prefill_dq_gemm_engaged;
    const routed = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(routed);
    try testing.expectEqual(before + 1, prefill_dq_gemm_engaged);
    const routed_f = try evalToF32(al, routed, on, s);
    defer al.free(routed_f);

    var stock_err: f32 = 0;
    var routed_err: f32 = 0;
    for (0..on) |i| {
        stock_err = @max(stock_err, @abs(stock_f[i] - gtd[i]));
        routed_err = @max(routed_err, @abs(routed_f[i] - gtd[i]));
    }
    // bf16-weight-rounding is the only extra error source — same magnitude
    // class as the stock kernel's own bf16 output rounding.
    try testing.expect(routed_err <= 1.5 * stock_err + 5e-3);

    // Decode/verify widths never route (M below the floor).
    var x8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x8);
    {
        var x8_32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x8_32);
        const s8 = [_]c_int{ 0, 0, 0 };
        const e8 = [_]c_int{ 1, 8, K };
        const st8 = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&x8_32, x32, &s8, 3, &e8, 3, &st8, 3, s));
        try mlx.check(mlx.mlx_astype(&x8, x8_32, .bfloat16, s));
    }
    const small_before = prefill_dq_gemm_engaged;
    const small = try qmatmulBits(x8, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(small);
    try mlx.check(mlx.mlx_array_eval(small));
    try testing.expectEqual(small_before, prefill_dq_gemm_engaged);
}

test "qmatmulBits nvfp4 matches dequantize+matmul reference" {
    // NVFP4 (issue #24): packed-u32 weight + fp8-e4m3 uint8 scales, NO biases,
    // group_size 16. qmatmulBits must route the bias-less weight to mode
    // "nvfp4" — the legacy null-bias heuristic would misroute it to "mxfp8"
    // (bits=8) and produce garbage.
    const s = mlx.gpuStream();

    const OUT = 8;
    const IN = 64; // multiple of the nvfp4 group size (16)
    var w_host: [OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5))) * 0.07;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 2, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize with mode=nvfp4 → vector [q, scales] (no biases element).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));

    const null_bi = mlx.mlx_array{ .ctx = null };

    // Reference: dequantize(mode=nvfp4) → x @ wᵀ in plain matmul.
    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, null_bi, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var w_deq_host: [OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    var x_host: [IN]f32 = undefined;
    for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
    var x_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_f32);
    {
        const sh = [_]c_int{ 1, IN };
        x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 2, .float32);
    }
    var x_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_bf);
    try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
    var x_bf_host: [IN]f32 = undefined;
    try testReadF32(x_bf, &x_bf_host, s);

    // Ground truth from the dequantized weights.
    var gt: [OUT]f32 = undefined;
    for (0..OUT) |o| {
        var acc: f32 = 0;
        for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[o * IN + in];
        gt[o] = acc;
    }

    const got = try qmatmulBits(x_bf, q_w, q_sc, null_bi, 4, 16, .nvfp4, s);
    defer _ = mlx.mlx_array_free(got);
    var got_host: [OUT]f32 = undefined;
    try testReadF32(got, &got_host, s);

    for (0..OUT) |o| try testing.expectApproxEqAbs(gt[o], got_host[o], 5e-2);
}

test "gatherQmv is no worse than stock gather_qmm vs fp32 dequant ground truth" {
    // Per the kernel-testing rule: assert NO-WORSE-THAN-REFERENCE against fp32
    // dequant ground truth, never kernel-vs-kernel agreement (both round to
    // bf16 in different orders, so exact agreement is not a real invariant).
    // Covers BOTH input layouts: the shared-x gate/up projection and the
    // per-expert-x down projection.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rnd = prng.random();

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 2, .gs = 64 },
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 8, .gs = 32 },
    }) |cfg| {
        // fp32 is the LIVE Laguna case: expert_x arrives as float32 while the
        // bank's scales/biases are bf16. A bf16-only test missed it entirely.
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
        for ([_]bool{ false, true }) |x_per_expert| {
            // ── quantized bank [E,N,K] ──
            const wcnt: usize = @intCast(E * N * K);
            const wbuf = try allocator.alloc(f32, wcnt);
            defer allocator.free(wbuf);
            for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const wsh = [_]c_int{ E, N, K };
            const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wsh, 3, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            var wsc = mlx.mlx_array_new();
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            defer _ = mlx.mlx_array_free(wsc);
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            // ── x: [K] shared, or [TOPK,K] per expert ──
            const xrows: c_int = if (x_per_expert) TOPK else 1;
            const xcnt: usize = @intCast(xrows * K);
            const xbuf = try allocator.alloc(f32, xcnt);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xsh = [_]c_int{ xrows, K };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, xdt, s));

            const idxvals = [_]u32{ 5, 0, 3 };
            const idxsh = [_]c_int{TOPK};
            const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 1, .uint32);
            defer _ = mlx.mlx_array_free(inds);

            // ── fp32 ground truth: dequantize the bank, take the 3 experts,
            // and do the matmul entirely in float32.
            var deq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq);
            try mlx.check(mlx.mlx_dequantize(&deq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, .{}, s));
            var deq32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq32);
            try mlx.check(mlx.mlx_astype(&deq32, deq, .float32, s));
            var sel = mlx.mlx_array_new(); // [TOPK,N,K]
            defer _ = mlx.mlx_array_free(sel);
            try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
            var selT = mlx.mlx_array_new(); // [TOPK,K,N]
            defer _ = mlx.mlx_array_free(selT);
            const tax = [_]c_int{ 0, 2, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&selT, sel, &tax, 3, s));
            // x as [TOPK,1,K] (broadcast the shared row when needed)
            var x32r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32r);
            const x3 = [_]c_int{ xrows, 1, K };
            try mlx.check(mlx.mlx_reshape(&x32r, x32, &x3, 3, s));
            var x32b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32b);
            const xb3 = [_]c_int{ TOPK, 1, K };
            try mlx.check(mlx.mlx_broadcast_to(&x32b, x32r, &xb3, 3, s));
            var truth = mlx.mlx_array_new(); // [TOPK,1,N]
            defer _ = mlx.mlx_array_free(truth);
            try mlx.check(mlx.mlx_matmul(&truth, x32b, selT, s));
            const tflat = [_]c_int{ TOPK, N };
            var truth2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(truth2);
            try mlx.check(mlx.mlx_reshape(&truth2, truth, &tflat, 2, s));

            // ── stock gather_qmm, same layouts ──
            var xg = mlx.mlx_array_new(); // [TOPK,1,K]
            defer _ = mlx.mlx_array_free(xg);
            {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_reshape(&xr, x, &x3, 3, s));
                try mlx.check(mlx.mlx_broadcast_to(&xg, xr, &xb3, 3, s));
            }
            var stock3 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stock3);
            const lhs3 = [_]u32{ 0, 1, 2 };
            const lhs_used = if (x_per_expert) blk: {
                break :blk mlx.mlx_array_new_data(&lhs3, &idxsh, 1, .uint32);
            } else blk: {
                const zeros = [_]u32{ 0, 0, 0 };
                break :blk mlx.mlx_array_new_data(&zeros, &idxsh, 1, .uint32);
            };
            defer _ = mlx.mlx_array_free(lhs_used);
            try mlx.check(mlx.mlx_gather_qmm(&stock3, xg, wq, wsc, wbi, lhs_used, inds, true, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", false, s));
            var stock2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stock2);
            try mlx.check(mlx.mlx_reshape(&stock2, stock3, &tflat, 2, s));

            // ── ours ──
            const maybe = try gatherQmv(s, x, wq, wsc, wbi, inds, cfg.bits, cfg.gs, .affine, x_per_expert);
            try testing.expect(maybe != null);
            const ours = maybe.?;
            defer _ = mlx.mlx_array_free(ours);

            const maxAbsErr = struct {
                fn go(a: mlx.mlx_array, ref: mlx.mlx_array, str: mlx.mlx_stream) !f32 {
                    var a32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(a32);
                    try mlx.check(mlx.mlx_astype(&a32, a, .float32, str));
                    var d = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(d);
                    try mlx.check(mlx.mlx_subtract(&d, a32, ref, str));
                    var ad = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(ad);
                    try mlx.check(mlx.mlx_abs(&ad, d, str));
                    var mx = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(mx);
                    try mlx.check(mlx.mlx_max(&mx, ad, false, str));
                    try mlx.check(mlx.mlx_array_eval(mx));
                    var out: f32 = 0;
                    try mlx.check(mlx.mlx_array_item_float32(&out, mx));
                    return out;
                }
            }.go;

            const err_ours = try maxAbsErr(ours, truth2, s);
            const err_stock = try maxAbsErr(stock2, truth2, s);
            // No worse than stock against the same fp32 ground truth (small
            // slack for bf16 accumulation-order differences).
            testing.expect(err_ours <= err_stock * 1.05 + 1e-3) catch |e| {
                std.debug.print("\n[gather-qmv] bits={d} gs={d} xdt={any} per_expert_x={} ours_err={d:.6} stock_err={d:.6}\n", .{ cfg.bits, cfg.gs, xdt, x_per_expert, err_ours, err_stock });
                return e;
            };
        }
        }
    }
}

test "gatherExpertMm nvfp4 matches per-expert dequantized reference" {
    // MoE expert dispatch on an nvfp4 checkpoint: gather_qmm must receive
    // mode="nvfp4" + null biases (decode AND prefill shapes, same calling
    // convention as moeMLP2).
    const s = mlx.gpuStream();

    const E = 4;
    const IN = 32;
    const OUT = 8;

    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));

    const null_bi = mlx.mlx_array{ .ctx = null };
    const no_idx = mlx.mlx_array{ .ctx = null };

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, null_bi, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, null_bi, no_idx, inds, 4, 16, .nvfp4, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s)); // [K, OUT]
        var quant_host: [K * OUT]f32 = undefined;
        try testReadF32(quant, &quant_host, s);

        for (0..K * OUT) |i| try testing.expectApproxEqAbs(gt[i], quant_host[i], 5e-2);
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 };
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, null_bi, no_idx, sorted_inds, 4, 16, .nvfp4, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        var quant_host: [N * OUT]f32 = undefined;
        try testReadF32(quant, &quant_host, s);

        for (0..N * OUT) |i| try testing.expectApproxEqAbs(gt[i], quant_host[i], 5e-2);
    }
}

test "appendLinearAttnWeights skips fields with null ctx (plain bf16 layers)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for the 9 non-scale/bias fields so they have non-null ctx.
    const sh = [_]c_int{1};
    var arrs: [9]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // Simulate the UD layout: weights set, scales/biases null.
    const la: LinearAttnWeights = .{
        .combined_proj = false,
        .qkv_w = arrs[0],
        .qkv_s = mlx.mlx_array{ .ctx = null },
        .qkv_b = mlx.mlx_array{ .ctx = null },
        .z_w = arrs[1],
        .z_s = mlx.mlx_array{ .ctx = null },
        .z_b = mlx.mlx_array{ .ctx = null },
        .a_w = arrs[2],
        .a_s = mlx.mlx_array{ .ctx = null },
        .a_b = mlx.mlx_array{ .ctx = null },
        .b_w = arrs[3],
        .b_s = mlx.mlx_array{ .ctx = null },
        .b_b = mlx.mlx_array{ .ctx = null },
        .conv1d_w = arrs[4],
        .A_log = arrs[5],
        .dt_bias = arrs[6],
        .norm_w = arrs[7],
        .out_w = arrs[8],
        .out_s = mlx.mlx_array{ .ctx = null },
        .out_b = mlx.mlx_array{ .ctx = null },
    };

    appendLinearAttnWeights(vec, &la);

    // Of 19 mlx_array fields: 5 weights (qkv/z/a/b/out) + 4 SSM bits
    // (conv1d/A_log/dt_bias/norm_w) = 9 expected. The 10 null-ctx scales/biases
    // are skipped — confirms the optional-bf16 path doesn't poison the eval batch.
    try testing.expectEqual(@as(usize, 9), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips MoE fields with null ctx (UD MoE bf16 router/SEG)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for every weight (`*_w`) — non-null ctx. Quantized projections
    // also have real scales/biases. UD bf16 layers (router, shared_expert_gate)
    // get null-ctx scales/biases.
    const sh = [_]c_int{1};
    var arrs: [16]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // UD MoE Qwen3.5 layout: router + shared_expert_gate are bf16 (null s/b);
    // routed experts (switch_*) and shared_expert (shared_*) stay quantized.
    const mw: MoeMlpWeights = .{
        .router_w = arrs[0],
        .router_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .router_b = mlx.mlx_array{ .ctx = null }, // UD bf16
        .switch_gate_w = arrs[1],
        .switch_gate_s = arrs[2],
        .switch_gate_b = arrs[3],
        .switch_up_w = arrs[4],
        .switch_up_s = arrs[5],
        .switch_up_b = arrs[6],
        .switch_down_w = arrs[7],
        .switch_down_s = arrs[8],
        .switch_down_b = arrs[9],
        .shared_gate_w = arrs[10],
        .shared_gate_s = arrs[11],
        .shared_gate_b = arrs[12],
        .shared_up_w = arrs[13],
        .shared_up_s = arrs[14],
        .shared_up_b = arrs[15],
        .shared_down_w = arrs[0], // reuse — only ctx-null check matters here
        .shared_down_s = arrs[1],
        .shared_down_b = arrs[2],
        .shared_expert_gate_w = arrs[3],
        .shared_expert_gate_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .shared_expert_gate_b = mlx.mlx_array{ .ctx = null }, // UD bf16
        .router_scale = null, // None — Qwen3.5 doesn't use sigma-MoE
        .per_expert_scale = null,
    };
    const hw: HybridMlpWeights = .{ .moe = mw };

    appendHybridMlpWeights(vec, &hw);

    // Counted by hand: 21 non-optional `mlx.mlx_array` fields, of which 2 are
    // null-ctx (router_s, router_b) → 19 appended. Plus the 5 optional
    // `?mlx.mlx_array` fields: shared_expert_gate_w is Some(real) → +1; SEG
    // scales/biases are Some(null-ctx) → +0 each; router_scale and
    // per_expert_scale are None → +0 each. Total: 19 + 1 = 20.
    try testing.expectEqual(@as(usize, 20), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips dense fields with null ctx (UD dense bf16)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    const sh = [_]c_int{1};
    var arrs: [3]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // All-bf16 dense MLP: weights set, scales/biases null.
    const dw: DenseMlpWeights = .{
        .gate_w = arrs[0],
        .gate_s = mlx.mlx_array{ .ctx = null },
        .gate_b = mlx.mlx_array{ .ctx = null },
        .up_w = arrs[1],
        .up_s = mlx.mlx_array{ .ctx = null },
        .up_b = mlx.mlx_array{ .ctx = null },
        .down_w = arrs[2],
        .down_s = mlx.mlx_array{ .ctx = null },
        .down_b = mlx.mlx_array{ .ctx = null },
    };
    const hw: HybridMlpWeights = .{ .dense = dw };

    appendHybridMlpWeights(vec, &hw);

    // 9 fields, 3 weights non-null + 6 null-ctx scales/biases skipped → 3.
    try testing.expectEqual(@as(usize, 3), mlx.mlx_vector_array_size(vec));
}

test "batchedExpertDecodePolicy: batched decode is a validated per-arch opt-in" {
    // The batchedExpertMm decode path (take + batched quantized_matmul) dodges
    // our self-built libmlx's serialized decode gather_qmm. It is a big win only
    // where the per-token gather cost dominates (Laguna 2-bit large experts:
    // 17→48 tok/s) and a NET LOSS where experts are small and the take-
    // materialization overhead dominates (gemma4-26B-A4B: 114→85, Qwen3.6-MoE
    // likewise — captured in the 26.7.10 bench). So it must NEVER be default-on
    // for all quantized MoE — only Laguna opts in by default.
    try testing.expect(batchedExpertDecodePolicy("laguna", false, false));
    try testing.expect(!batchedExpertDecodePolicy("gemma4_text", false, false));
    try testing.expect(!batchedExpertDecodePolicy("qwen3_5_moe", false, false));
    try testing.expect(!batchedExpertDecodePolicy("hy_v3", false, false));
    // MLX_SERVE_MOE_BATCHED_DECODE forces it on for any arch (experimentation A/B).
    try testing.expect(batchedExpertDecodePolicy("gemma4_text", false, true));
    // MLX_SERVE_MOE_GATHER_DECODE is the hard override: beats both the laguna
    // default and the batched force.
    try testing.expect(!batchedExpertDecodePolicy("laguna", true, false));
    try testing.expect(!batchedExpertDecodePolicy("gemma4_text", true, true));
}

test "moeRoutingChain produces top-K indices and renormalized softmax weights" {
    const s = mlx.gpuStream();

    // Two rows of router logits over 6 experts. Top-2 of each row is unambiguous:
    //   row 0: experts {0, 3} (logits 10 and 5)
    //   row 1: experts {1, 4} (logits 10 and 5)
    const n_rows: c_int = 2;
    const n_exp: c_int = 6;
    const k: c_int = 2;
    const data = [_]f32{
        10.0, 0.0,  0.0, 5.0, 0.0, 0.0,
        0.0,  10.0, 0.0, 0.0, 5.0, 0.0,
    };
    const shape = [_]c_int{ n_rows, n_exp };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const routed = try moeRoutingChain(logits, k, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Shape check (cheap, no data read needed).
    {
        const inds_shape = mlx.getShape(routed.inds);
        try testing.expectEqual(@as(usize, 2), inds_shape.len);
        try testing.expectEqual(n_rows, inds_shape[0]);
        try testing.expectEqual(k, inds_shape[1]);
        const sc_shape = mlx.getShape(routed.norm_scores);
        try testing.expectEqual(@as(usize, 2), sc_shape.len);
        try testing.expectEqual(n_rows, sc_shape[0]);
        try testing.expectEqual(k, sc_shape[1]);
    }

    // To verify top-K correctness without reading non-contiguous slice memory
    // directly, gather the original logits at the selected indices: gathered[i,j]
    // == logits[i, inds[i,j]]. Then sum across K — for our fixture, the top-2
    // logits in each row are {10, 5}, so the per-row sum must be 15.
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, logits, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // norm_scores must sum to 1 along K (verifies the renormalize step).
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, routed.norm_scores, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    // gathered_sum and scores_sum are 1D outputs of sum_axis (contiguous).
    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    const tol: f32 = 1e-3;
    try testing.expect(@abs(gs[0] - 15.0) < tol);
    try testing.expect(@abs(gs[1] - 15.0) < tol);
    try testing.expect(@abs(ss[0] - 1.0) < tol);
    try testing.expect(@abs(ss[1] - 1.0) < tol);
}

test "hy3RoutingChain selects on biased scores but weights by original sigmoid scores" {
    const s = mlx.gpuStream();

    // 1 row over 6 experts, k=2. sigmoid(logits):
    //   [0.8808, 0.5, 0.1192, 0.7311, 0.2689, 0.6225]
    // With NO bias the top-2 would be {0 (0.8808), 3 (0.7311)}. The expert bias
    // +0.5 on expert 1 lifts it to 1.0, so selection must pick {1, 0} — but the
    // WEIGHTS must come from the ORIGINAL sigmoid scores {0.5, 0.8808}
    // (sum 1.3808), renormalized to 1 then scaled by router_scaling_factor.
    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const bias_data = [_]f32{ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const scale: f32 = 2.0;
    const routed = try hy3RoutingChain(logits, bias, 2, true, scale, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Gather the ORIGINAL sigmoid scores at the selected indices. If selection
    // ignored the bias the gathered sum would be 0.8808+0.7311 = 1.6119; with
    // bias-driven selection it must be 0.5+0.8808 = 1.3808.
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logits, s));
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, sig, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // Renorm + scale: scores sum to exactly router_scaling_factor.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // norm_scores are bf16 (cast for the expert-combine multiply) → ~0.4% rel.
    try testing.expect(@abs(gs[0] - 1.3808) < 1e-3);
    try testing.expect(@abs(ss[0] - scale) < 0.02);

    // Per-element weights pin "weighted by ORIGINAL scores": {0.5, 0.8808}
    // renormed ×2 → {0.7242, 1.2758} (any order). Had the chain weighted by
    // the BIASED scores {1.0, 0.8808} the pair would be {1.0634, 0.9366} —
    // same sum, different elements, so the sum check alone can't see it.
    try mlx.check(mlx.mlx_array_eval(scores_f32));
    const sv = mlx.mlx_array_data_float32(scores_f32) orelse return error.InvalidDtype;
    const lo = @min(sv[0], sv[1]);
    const hi = @max(sv[0], sv[1]);
    try testing.expect(@abs(lo - 0.7242) < 0.02);
    try testing.expect(@abs(hi - 1.2758) < 0.02);
}

test "hy3RoutingChain route_norm=false keeps raw sigmoid weights (scaled only)" {
    const s = mlx.gpuStream();

    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Zero bias: selection = top-2 of raw sigmoid = {0.8808, 0.7311}.
    const bias_data: [6]f32 = @splat(0.0);
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const routed = try hy3RoutingChain(logits, bias, 2, false, 2.0, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));
    try mlx.check(mlx.mlx_array_eval(scores_sum));

    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // (0.880797 + 0.731059) × 2.0 = 3.2237 — un-normalized, scaled.
    try testing.expect(@abs(ss[0] - 3.2237) < 0.02);
}

test "computeYarnFreqs matches HF _compute_yarn_parameters (Laguna full-attn rope)" {
    // Golden values from the reference YaRN math (tests/dump_laguna_fixtures.py
    // and the pure-Python cross-check): head_dim 128, partial 0.5 → dim 64 →
    // 32 freqs; base 5e5, factor 32, beta_fast 32, beta_slow 1, orig_max 8192.
    // low/high correction dims truncate to 9/18.
    var freqs: [32]f64 = undefined;
    computeYarnFreqs(&freqs, 128, 0.5, 500000.0, 32.0, 32.0, 1.0, 8192);
    // Spot-check the denominator array mlx_fast_rope consumes (angle = pos/freqs).
    const golden = [_]struct { idx: usize, val: f64 }{
        .{ .idx = 0, .val = 1.000000000e+00 },
        .{ .idx = 1, .val = 1.506929076e+00 },
        .{ .idx = 2, .val = 2.270835240e+00 },
        .{ .idx = 7, .val = 1.764613870e+01 },
        .{ .idx = 15, .val = 1.324904288e+03 },
        .{ .idx = 16, .val = 2.868264127e+03 },
        .{ .idx = 23, .val = 3.992865387e+05 },
        .{ .idx = 31, .val = 1.061761980e+07 },
    };
    for (golden) |g| {
        try testing.expectApproxEqRel(g.val, freqs[g.idx], 1e-7);
    }
    // Below the low correction dim (9): pure extrapolation → freqs = base^(2i/64).
    try testing.expectApproxEqRel(std.math.pow(f64, 500000.0, 0.0), freqs[0], 1e-9);
    // Above the high correction dim (18): pure interpolation → freqs = factor·base^(2i/64).
    try testing.expectApproxEqRel(32.0 * std.math.pow(f64, 500000.0, @as(f64, 2 * 31) / 64.0), freqs[31], 1e-6);
}

test "laguna yarn parity vs modeling_laguna.py (LAGUNA_FIXTURES)" {
    // Env-gated cos/value oracle: compare computeYarnFreqs against the
    // full-attention rotary frequencies dumped from the reference
    // modeling_laguna.py (tests/dump_laguna_fixtures.py, CPU fp32). Dormant in
    // normal CI; run with LAGUNA_FIXTURES=/tmp/laguna_fixtures.json. The
    // fixture's `yarn.freqs` = 1/inv_freq (the mlx_fast_rope denominator).
    const path_z = std.c.getenv("LAGUNA_FIXTURES") orelse return error.SkipZigTest;
    const path = std.mem.span(path_z);
    if (path.len == 0) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const data = try reader_state.interface.allocRemaining(testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(data);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, data, .{});
    defer parsed.deinit();
    const yarn = parsed.value.object.get("yarn").?.object;
    const cfg = parsed.value.object.get("config").?.object;

    const head_dim: u32 = @intCast(cfg.get("head_dim").?.integer);
    const partial: f32 = @floatCast(jsonF64(yarn.get("partial_rotary_factor").?));
    const base = jsonF64(yarn.get("rope_theta").?);
    const factor = jsonF64(yarn.get("factor").?);
    const beta_fast = jsonF64(yarn.get("beta_fast").?);
    const beta_slow = jsonF64(yarn.get("beta_slow").?);
    const orig_max: u32 = @intCast(yarn.get("original_max_position_embeddings").?.integer);

    const ref = yarn.get("freqs").?.array;
    const out = try testing.allocator.alloc(f64, ref.items.len);
    defer testing.allocator.free(out);
    computeYarnFreqs(out, head_dim, partial, base, factor, beta_fast, beta_slow, orig_max);
    for (ref.items, 0..) |item, i| {
        try testing.expectApproxEqRel(jsonF64(item), out[i], 1e-6);
    }
    std.debug.print("[laguna-yarn] {d} freqs match reference (max rel err < 1e-6)\n", .{ref.items.len});
}

fn jsonF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

test "gdnGateChain matches g = exp(-exp(A_log) * softplus(a + dt_bias))" {
    const s = mlx.gpuStream();

    // Hv = 2 heads, B=1, S=1. Hand-computable fixtures:
    //   head 0: A_log=0, a=0, dt_bias=0 → g = exp(-1 * softplus(0)) = exp(-ln2) = 0.5
    //   head 1: A_log=ln2, a=0, dt_bias=0 → g = exp(-2 * ln2) = 0.25
    const a_log_data = [_]f32{ 0.0, @log(2.0) };
    const hv_shape = [_]c_int{2};
    const A_log = mlx.mlx_array_new_data(&a_log_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(A_log);

    const a_data = [_]f32{ 0.0, 0.0 };
    const a_shape = [_]c_int{ 1, 1, 2 };
    const a = mlx.mlx_array_new_data(&a_data, &a_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(a);

    const dt_data = [_]f32{ 0.0, 0.0 };
    const dt_bias = mlx.mlx_array_new_data(&dt_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = try gdnGateChain(A_log, a, dt_bias, s);
    defer _ = mlx.mlx_array_free(g);

    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_astype(&g_f32, g, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g_f32));

    const gd = mlx.mlx_array_data_float32(g_f32) orelse return error.InvalidDtype;
    // bf16 round-trip tolerance
    const tol: f32 = 5e-3;
    try testing.expect(@abs(gd[0] - 0.5) < tol);
    try testing.expect(@abs(gd[1] - 0.25) < tol);
}

// Helper for the GDN seq-kernel parity test below: run the recurrence via the
// requested kernel variant and return the FINAL state [B,Hv,Dv,Dk] as f32.
fn gdnTestRun(comptime seq: bool, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
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
        const kern = try getGdnKernelSeq();
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
        const kern = try getGdnKernel();
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 1));
        return out;
    }
}

test "GDN seq-kernel final state matches single-state kernel (PLD capture parity)" {
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

    const state_a = try gdnTestRun(false, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_a);
    const state_b = try gdnTestRun(true, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
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

    const ref2 = try gdnTestRun(false, q2, k2, v2, g2, b2, st, B, acc_run, Hk, Hv, Dk, Dv, s);
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));
    const seq_stride = mlx.mlx_array_new_int(B * Hv * Dv * Dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar, seq_stride };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try getGdnKernelSeq();
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
fn bf16Trunc(x: f32) f32 {
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

/// Run the GDN recurrence via the stock single-state kernel (blocked=false)
/// or the blocked-seq prefill kernel (blocked=true) and return BOTH outputs.
fn gdnRunYState(blocked: bool, tb: u32, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !GdnRunOut {
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    if (blocked) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256 * @divExact(Dv, 32), Hv, B));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, Dv, B * Hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = if (blocked) try getGdnKernelBlocked(tb) else try getGdnKernel();
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

/// Eval an mlx array as f32 and copy its data into a fresh slice.
fn evalToF32(al: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.InvalidDtype;
    const out = try al.alloc(f32, n);
    @memcpy(out, d[0..n]);
    return out;
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

const GdnCase = struct { B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dv: c_int, tb: u32 };

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
    try mlx.check(mlx.mlx_astype(&q, q32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&v, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&st, st32, .bfloat16, s));

    const stock = try gdnRunYState(false, 0, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(stock.y);
    defer _ = mlx.mlx_array_free(stock.state);
    const blocked = try gdnRunYState(true, case.tb, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(blocked.y);
    defer _ = mlx.mlx_array_free(blocked.state);

    const stock_y = try evalToF32(al, stock.y, vn, s);
    defer al.free(stock_y);
    const stock_st = try evalToF32(al, stock.state, sn, s);
    defer al.free(stock_st);
    const blk_y = try evalToF32(al, blocked.y, vn, s);
    defer al.free(blk_y);
    const blk_st = try evalToF32(al, blocked.state, sn, s);
    defer al.free(blk_st);

    // No-worse-than-reference: the blocked kernel's error vs the f64 ground
    // truth must not exceed the stock kernel's (both bf16-out; 1.5x headroom
    // covers fp32 summation-order differences, gross bugs blow way past it).
    const stock_y_err = maxAbsDiff(stock_y, ref.y);
    const stock_st_err = maxAbsDiff(stock_st, ref.state);
    const blk_y_err = maxAbsDiff(blk_y, ref.y);
    const blk_st_err = maxAbsDiff(blk_st, ref.state);
    if (blk_y_err > 1.5 * stock_y_err + 0.02 or blk_st_err > 1.5 * stock_st_err + 0.02) {
        std.debug.print(
            "GDN blocked parity FAIL (B={d} T={d} Hk={d} Hv={d} Dv={d} tb={d}): y {d:.5} vs stock {d:.5}, state {d:.5} vs stock {d:.5}\n",
            .{ case.B, case.T, case.Hk, case.Hv, case.Dv, case.tb, blk_y_err, stock_y_err, blk_st_err, stock_st_err },
        );
        return error.GdnBlockedParityFailed;
    }
}

test "GDN blocked-seq kernel: no worse than stock vs f64 ground truth (T/GQA/Dv/TB sweep)" {
    const s = mlx.gpuStream();
    // T deliberately hits non-multiples of every supported TB (16/32/48);
    // Hk<Hv exercises GQA head mapping; Dv 32..128 exercises 1..4 DB blocks.
    const cases = [_]GdnCase{
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32 }, // live-like 27B geometry (fewer heads)
        .{ .B = 2, .T = 64, .Hk = 1, .Hv = 2, .Dv = 32, .tb = 32 }, // batch>1, minimal Dv
        .{ .B = 1, .T = 65, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16 }, // TB=16 variant, ragged tail 1
        .{ .B = 1, .T = 49, .Hk = 2, .Hv = 2, .Dv = 32, .tb = 48 }, // TB=48 variant, ragged tail 1
    };
    for (cases) |case| try gdnBlockedParityCase(case, s);
}

test "GDN blocked-seq kernel: chunk-boundary state continuity (split run == full run)" {
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
    const full = try gdnRunYState(true, 32, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
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

    const part1 = try gdnRunYState(true, 32, q1, k1, v1, g1, b1, st, B, T1, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(part1.y);
    defer _ = mlx.mlx_array_free(part1.state);
    const part2 = try gdnRunYState(true, 32, q2, k2, v2, g2, b2, part1.state, B, T - T1, Hk, Hv, Dk, Dv, s);
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

/// Assert a verify-kernel output is NO LESS ACCURATE than the stock qmm it
/// replaces, both measured against `wdq_t` — the SAME 4-bit weights dequantized
/// to fp32 (so quantization error cancels and only the two kernels' own
/// arithmetic is under test).
///
/// This is the machine-independent invariant. A direct kernel-vs-stock bound is
/// NOT: both paths accumulate fp32 in a different order and round to bf16, so on
/// heavily-cancelling data each already sits ~0.03 from truth on its worst
/// element and they may legitimately differ from each other by more than that.
/// Tuning a pair-agreement threshold on one GPU produces a test that is green on
/// the dev machine and red on the CI runner while the kernel is bit-for-bit as
/// accurate (measured: kernel-vs-truth == stock-vs-truth to four decimals).
/// A genuine defect — a partial-sum race, a register spill, bad indexing —
/// pushes kernel-vs-truth far past stock-vs-truth and is still caught here.
fn expectVerifyQmmNoWorseThanStock(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    wq: mlx.mlx_array,
    wsc: mlx.mlx_array,
    wbi: mlx.mlx_array,
    bits: u32,
    gs: u32,
    wdq_t: mlx.mlx_array,
    got: mlx.mlx_array,
    label: []const u8,
) !void {
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_quantized_matmul(&ref, x, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));

    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wdq_t, s));

    var tf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tf);
    var rf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf);
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&tf, truth, .float32, s));
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(tf));
    try mlx.check(mlx.mlx_array_eval(rf));
    try mlx.check(mlx.mlx_array_eval(gf));
    const td = mlx.mlx_array_data_float32(tf).?;
    const rd = mlx.mlx_array_data_float32(rf).?;
    const gd = mlx.mlx_array_data_float32(gf).?;

    const gsh = mlx.getShape(got);
    const count: usize = @intCast(gsh[1] * gsh[2]);
    var stock_max: f64 = 0;
    var kern_max: f64 = 0;
    var dot_s: f64 = 0;
    var dot_k: f64 = 0;
    var nt: f64 = 0;
    var ns: f64 = 0;
    var nk: f64 = 0;
    for (0..count) |i| {
        const t: f64 = td[i];
        const r: f64 = rd[i];
        const g: f64 = gd[i];
        // Clamped denominator: a near-zero output of a heavily-cancelling dot
        // product must not manufacture a huge ratio from a tiny absolute error.
        const denom = @max(1.0, @abs(t));
        stock_max = @max(stock_max, @abs(r - t) / denom);
        kern_max = @max(kern_max, @abs(g - t) / denom);
        dot_s += r * t;
        dot_k += g * t;
        nt += t * t;
        ns += r * r;
        nk += g * g;
    }
    const cos_stock = dot_s / (@sqrt(ns) * @sqrt(nt));
    const cos_kern = dot_k / (@sqrt(nk) * @sqrt(nt));

    // The kernel may not be materially less accurate than stock…
    if (kern_max > stock_max + 0.01 or cos_kern < cos_stock - 1e-5) {
        std.debug.print(
            "verifyQmm LESS ACCURATE than stock [{s}]: kernel_vs_truth={d:.4} (cos {d:.6}) stock_vs_truth={d:.4} (cos {d:.6})\n",
            .{ label, kern_max, cos_kern, stock_max, cos_stock },
        );
        return error.TestExpectedApproxEq;
    }
    // …and both must actually be tracking the truth (catches a broken reference
    // or a kernel that fails in the same direction stock does).
    if (cos_kern < 0.999) {
        std.debug.print("verifyQmm not tracking fp32 truth [{s}]: cos={d:.6}\n", .{ label, cos_kern });
        return error.TestExpectedApproxEq;
    }
}

test "verifyQmm: split-K + msg + NAX verify-width kernels match stock qmm (4-bit affine)" {
    // The spec-verify fast path: stock MLX qmm is tuned for M=1 (qmv) and
    // large M (steel); the 2..8-row verify shapes fall in a dead zone.
    // Port of MTPLX's split-K verify kernel family (verify_kernels.py,
    // Apache-2.0) — parity vs mlx_quantized_matmul at real 27B shapes.
    //
    // THE INVARIANT IS "NO LESS ACCURATE THAN STOCK", NOT "AGREES WITH STOCK".
    // Both paths accumulate in fp32 in a DIFFERENT ORDER and then round the
    // result to bf16, so the two outputs legitimately disagree by a few ULPs —
    // and on this data (K=5120 dot products of U(-0.5,0.5) → heavy cancellation)
    // each path already sits up to ~0.036 from the true value on its worst
    // element, purely from that bf16 output rounding. A direct kernel-vs-stock
    // bound therefore pins AGREEMENT tighter than bf16 permits: the original
    // `max_rel <= 0.02` / `cos > 0.99999` thresholds were tuned on one GPU and
    // the CI runner's GPU (different rounding, same correctness) failed them at
    // max_rel 0.0977 / cos 0.999988 — a green-here/red-there test that says
    // nothing about correctness. Measured: kernel-vs-truth equals stock-vs-truth
    // to four decimals on every shape.
    //
    // So both paths are measured against an fp32 DEQUANT REFERENCE (the same
    // 4-bit weights, dequantized, matmul'd in fp32 — the ground truth both are
    // approximating) and the kernel must be no worse than the stock kernel it
    // replaces. That is machine-independent, and a real defect (a partial-sum
    // race, a register spill, bad indexing) blows kernel-vs-truth far past
    // stock-vs-truth while a rounding-order difference cannot.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    const cases = [_]struct { k: c_int, n: c_int, gs: u32 }{
        .{ .k = 5120, .n = 17408, .gs = 64 }, // MLP gate/up
        .{ .k = 5120, .n = 1024, .gs = 32 }, // small-N above the gate, MTP gs-32 class
        .{ .k = 2048, .n = 5120, .gs = 128 }, // gs-128 arm
    };
    for (cases) |cs| {
        // Random dense weight → 4-bit affine triple.
        const wn: usize = @intCast(cs.n * cs.k);
        const wbuf = try allocator.alloc(f32, wn);
        defer allocator.free(wbuf);
        for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const wshape = [_]c_int{ cs.n, cs.k };
        const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        // fp32 ground truth operand: the same 4-bit weights, dequantized.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        // Kernel rows: split-K M 2..7 on every machine; the NAX m16 tile rows
        // (M 8..16, the padding path at 8/9/12 and the exact-16 tile) run only
        // where the M5-class probe is live (todo-m5-nax.md §9.3) — the probe
        // self-gates, so plain-SIMD machines skip them.
        const simd_ms = [_]c_int{ 2, 3, 4, 5, 6, 7 };
        const nax_ms = [_]c_int{ 8, 9, 12, 16 };
        const total_ms: usize = if (verifyQmmNaxEnabled()) simd_ms.len + nax_ms.len else simd_ms.len;
        var mi: usize = 0;
        while (mi < total_ms) : (mi += 1) {
            const m = if (mi < simd_ms.len) simd_ms[mi] else nax_ms[mi - simd_ms.len];
            const label: []const u8 = if (mi < simd_ms.len) "split-K" else "nax m16";
            const xn: usize = @intCast(m * cs.k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            // Kernel path — must ENGAGE for every listed M.
            const got_opt = try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs);
            try testing.expect(got_opt != null);
            const got = got_opt.?;
            defer _ = mlx.mlx_array_free(got);
            const gsh = mlx.getShape(got);
            try testing.expectEqual(@as(c_int, 1), gsh[0]);
            try testing.expectEqual(m, gsh[1]);
            try testing.expectEqual(cs.n, gsh[2]);

            try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, cs.gs, wdq_t, got, label);
        }

        // Ineligible widths fall through to stock (null). The NAX probe is
        // FORCED false so this pins the plain-SIMD (M1-M4) dispatch on every
        // machine, an M5 running the suite included: M=1 stays qmv, M 8..16
        // must NOT reach the split-K family (the measured spill cliff past 7
        // live row vectors), and M=17 is past every lane.
        {
            vqmm_nax_probe_override = false;
            defer vqmm_nax_probe_override = null;
            for ([_]c_int{ 1, 8, 16, 17 }) |bad_m| {
                const xn: usize = @intCast(bad_m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, bad_m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
            }
        }

        // M=1 stays stock under the REAL probe too (MTPLX's tile nominally
        // covers M 1..16, but M=1 belongs to stock qmv on every machine).
        {
            const xbuf = try allocator.alloc(f32, @intCast(cs.k));
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = 0.1;
            const xshape = [_]c_int{ 1, 1, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
        }
    }

    // oQe mixed-precision trunk projections: only the M5 NAX lane adopts
    // affine 5/6-bit weights. Keep this self-gated like the NAX rows above so
    // non-G17 CI never attempts to compile an unavailable MPP kernel.
    if (verifyQmmNaxEnabled()) {
        const k: c_int = 512;
        const n: c_int = 5120;
        const gs: u32 = 64;
        for ([_]u32{ 5, 6 }) |bits| {
            const wn: usize = @intCast(n * k);
            const wbuf = try allocator.alloc(f32, wn);
            defer allocator.free(wbuf);
            for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const wshape = [_]c_int{ n, k };
            const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(
                &triple,
                wb,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{},
                s,
            ));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(
                &wdq,
                wq,
                wsc,
                wbi,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{ .ctx = null },
                mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 },
                s,
            ));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            const m: c_int = 8;
            const xn: usize = @intCast(m * k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, gs);
            try testing.expect(got_opt != null);
            const got = got_opt.?;
            defer _ = mlx.mlx_array_free(got);
            try expectVerifyQmmNoWorseThanStock(
                s,
                x,
                wq,
                wsc,
                wbi,
                bits,
                gs,
                wdq_t,
                got,
                if (bits == 5) "nax oQe q5" else "nax oQe q6",
            );

            // Below the NAX takeover row these bit widths stay on stock qmm;
            // the existing split-K/msg kernels remain exact q4 specializations.
            const x6buf = try allocator.alloc(f32, 6 * @as(usize, @intCast(k)));
            defer allocator.free(x6buf);
            for (x6buf) |*v| v.* = 0.1;
            const x6shape = [_]c_int{ 1, 6, k };
            const x6f = mlx.mlx_array_new_data(x6buf.ptr, &x6shape, 3, .float32);
            defer _ = mlx.mlx_array_free(x6f);
            var x6 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x6);
            try mlx.check(mlx.mlx_astype(&x6, x6f, .bfloat16, s));
            try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x6, wq, wsc, wbi, bits, gs));
        }
    }

    // ── msg wide-tile parity (the huge-N/lm_head arm), called directly with
    // a RAGGED N (1000 % 32 != 0) so the in-kernel n0 guard is exercised. ──
    {
        const mk: c_int = 2048;
        const mn: c_int = 1000;
        const wn: usize = @intCast(mn * mk);
        const wbuf = try allocator.alloc(f32, wn);
        defer allocator.free(wbuf);
        for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const wshape = [_]c_int{ mn, mk };
        const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        var m: c_int = 2;
        while (m <= 6) : (m += 1) {
            const xn: usize = @intCast(m * mk);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, mk };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const xsh = mlx.getShape(x);
            const got = (try runVerifyQmmMsg(s, x, wq, wsc, wbi, 64, m, mk, mn, .bfloat16, xsh)).?;
            defer _ = mlx.mlx_array_free(got);

            try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, 64, wdq_t, got, "msg ragged-N");
        }
    }
}

test "vqmmLaneFor: NAX dispatch table (M 8..16 route to the m16 tile only when the probe is live)" {
    // Pure lane selection — todo-m5-nax.md §7. Hermetic on every machine:
    // no kernels are built, only the routing decision is pinned.
    const nax_off = false;
    const nax_on = true;

    // Plain-SIMD machine (probe false): the table is byte-identical to the
    // pre-NAX dispatch — M 8..16 fall through to stock.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 151936, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(16, 5120, 17408, nax_off, 8));
    // Geometry floors shared by every lane.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 256, nax_off, 8)); // N < 512
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 100, 17408, nax_off, 8)); // K % 64 != 0
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 17409, nax_off, 8)); // N % 4 != 0

    // M5 (probe true), default takeover width 8: M 2..7 KEEP the plain-SIMD
    // lanes (MTPLX's own dispatcher keeps SIMD through M=6 with NAX lit —
    // 16-row padding waste makes SIMD competitive at small M).
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(7, 5120, 151936, nax_on, 8));
    // M 8..16 route to the NAX tile when its stricter geometry holds…
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(8, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(16, 5120, 17408, nax_on, 8));
    // …including the lm_head class natively (N=151936 % 32 == 0 — no msg
    // variant needed past M=7).
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(9, 5120, 151936, nax_on, 8));
    // Tile-ineligible geometry at M 8..16 stays stock: K % 256, N % 32.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5184, 17408, nax_on, 8)); // 5184 % 256 == 64
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17412, nax_on, 8)); // 17412 % 32 == 4
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(17, 5120, 17408, nax_on, 8)); // past every lane
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_on, 8)); // M=1 stays qmv
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 480, nax_on, 8)); // tiny-N floor holds for NAX too

    // The M5-day A/B knob (MLX_SERVE_VERIFY_QMM_NAX_MIN_M=5): M 5..7 route
    // to NAX, M 4 keeps split-K, and the tile geometry is still required —
    // a lowered width never bypasses it, it falls back to the SIMD lanes.
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(5, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(7, 5120, 151936, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(5, 5184, 17408, nax_on, 5)); // K % 256 fails, % 64 holds
}

test "mixedNaxShapeEnabled keeps measured q5/q6 wins and rejects narrow regressions" {
    try testing.expect(mixedNaxShapeEnabled(4, 32, 1024));
    try testing.expect(mixedNaxShapeEnabled(5, 64, 5120));
    try testing.expect(mixedNaxShapeEnabled(6, 64, 5120));
    try testing.expect(!mixedNaxShapeEnabled(5, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(6, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(5, 32, 5120));
    try testing.expect(!mixedNaxShapeEnabled(6, 128, 5120));
}

test "oQ4e layer-role fingerprint pins every mixed override" {
    try testing.expectEqual(@as(usize, 30), std.mem.count(u8, OQE_MLP_DOWN_BITS, "5"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, OQE_MLP_DOWN_BITS, "6"));
    try testing.expectEqual(@as(usize, 11), std.mem.count(u8, OQE_LINEAR_Z_BITS, "5"));
    try testing.expectEqual(@as(usize, 20), std.mem.count(u8, OQE_LINEAR_AB_BITS, "5"));
    try testing.expectEqual(@as(usize, 48), std.mem.count(u8, OQE_LINEAR_OUT_BITS, "5"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, OQE_FULL_QK_BITS, "5"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, OQE_FULL_V_BITS, "6"));
    try testing.expectEqual(@as(usize, 3), std.mem.count(u8, OQE_FULL_O_BITS, "5"));
    try testing.expectEqual(@as(u32, 5), oqeLayerBits(OQE_MLP_DOWN_BITS, 0));
    try testing.expectEqual(@as(u32, 4), oqeLayerBits(OQE_MLP_DOWN_BITS, 63));
}

test "verifyQmmNaxEnabledForMFrom mirrors flags, min-M, and dispatch geometry" {
    const K: c_int = 5120;
    const N: c_int = 151936;

    try testing.expect(verifyQmmNaxEnabledForMFrom(8, K, N, true, true, true, 8));
    try testing.expect(verifyQmmNaxEnabledForMFrom(9, K, N, true, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, false, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, true, false, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, false, 8));

    // The takeover knob is part of controller readiness, not just kernel
    // dispatch: min-M 9 leaves M=8 on stock even though M=9 is NAX.
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, true, 9));
    try testing.expect(verifyQmmNaxEnabledForMFrom(9, K, N, true, true, true, 9));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(9, K, N, true, true, true, 10));
    try testing.expect(verifyQmmNaxEnabledForMFrom(8, K, N, true, true, true, 5));

    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, 5184, N, true, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, 151940, true, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(8, K, 480, true, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(1, K, N, true, true, true, 8));
    try testing.expect(!verifyQmmNaxEnabledForMFrom(17, K, N, true, true, true, 8));
}

fn mtpNaxTestConfig() ModelConfig {
    var config = ModelConfig{};
    config.model_type = "qwen3_5_moe";
    config.hidden_size = 5120;
    config.intermediate_size = 17408;
    config.num_hidden_layers = 64;
    config.vocab_size = 248320;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.full_attention_interval = 4;
    config.linear_num_key_heads = 16;
    config.linear_num_value_heads = 48;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.attn_output_gate = true;
    config.tie_word_embeddings = false;
    config.num_experts = 0;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.quant_mode = .affine;
    return config;
}

test "mtpNaxCalibratedModelFrom pins the complete measured dense Qwen3.6-27B architecture" {
    const good = mtpNaxTestConfig();
    try testing.expect(mtpNaxCalibratedModelFrom(&good, 248320));

    var bad = good;
    bad.model_type = "qwen3_moe";
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.hidden_size = 4096;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.intermediate_size = 17440;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_hidden_layers = 48;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.vocab_size = 151936;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 151936));
    try testing.expect(!mtpNaxCalibratedModelFrom(&good, 151936));
    bad = good;
    bad.num_attention_heads = 20;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_key_value_heads = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.head_dim = 128;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.full_attention_interval = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.linear_num_value_heads = 32;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.linear_value_head_dim = 64;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.attn_output_gate = false;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.tie_word_embeddings = true;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
    bad = good;
    bad.num_experts = 8;
    try testing.expect(!mtpNaxCalibratedModelFrom(&bad, 248320));
}

test "mtpNaxAffineProjectionMatches rejects mixed storage and off-profile quant geometry" {
    const s = mlx.gpuStream();
    const IN: u32 = 128;
    const OUT: u32 = 4;
    var config = ModelConfig{};
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.quant_mode = .affine;

    const mk = struct {
        fn arr(shape: []const c_int, dtype: mlx.mlx_dtype, stream: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dtype, stream));
            return a;
        }
    };

    const w4 = try mk.arr(&.{ OUT, IN * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w4);
    const sc64 = try mk.arr(&.{ OUT, IN / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sc64);
    const bi64 = try mk.arr(&.{ OUT, IN / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bi64);
    try testing.expect(mtpNaxAffineProjectionMatches(&config, w4, sc64, bi64, IN, OUT));
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc64, bi64, IN, OUT + 1));

    // Unsloth Dynamic projection: BF16 weight with no quant metadata.
    const w_bf16 = try mk.arr(&.{ OUT, IN }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(w_bf16);
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w_bf16, none, none, IN, OUT));

    // Same affine storage types, but per-tensor bits/group disagree with the
    // calibrated global 4-bit/gs-64 surface.
    const w2 = try mk.arr(&.{ OUT, IN * 2 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w2);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w2, sc64, bi64, IN, OUT));
    const sc128 = try mk.arr(&.{ OUT, IN / 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sc128);
    const bi128 = try mk.arr(&.{ OUT, IN / 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bi128);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc128, bi128, IN, OUT));

    // mxfp8-style uint8 scales cannot inherit the affine cost fit even if a
    // synthetic bias handle is present.
    const sc_mx = try mk.arr(&.{ OUT, IN / 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(sc_mx);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, w4, sc_mx, bi64, IN, OUT));

    // Material projections must also satisfy the actual M=8/M=9 NAX lane
    // geometry. K=256/N=512 qualifies; an otherwise valid K=320 affine
    // tensor does not (K % 256 != 0).
    const big_w = try mk.arr(&.{ 512, 256 * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(big_w);
    const big_sc = try mk.arr(&.{ 512, 256 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(big_sc);
    const big_bi = try mk.arr(&.{ 512, 256 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(big_bi);
    try testing.expect(mtpNaxAffineProjectionMatches(&config, big_w, big_sc, big_bi, 256, 512));

    const off_lane_w = try mk.arr(&.{ 512, 320 * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(off_lane_w);
    const off_lane_sc = try mk.arr(&.{ 512, 320 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(off_lane_sc);
    const off_lane_bi = try mk.arr(&.{ 512, 320 / 64 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(off_lane_bi);
    try testing.expect(!mtpNaxAffineProjectionMatches(&config, off_lane_w, off_lane_sc, off_lane_bi, 320, 512));
}

test "mtpNaxProfileEnabledFrom composes measured model, homogeneous trunk, and affine lm-head lane" {
    const model = mtpNaxTestConfig();
    const good: MtpNaxProfileInputs = .{
        .dense_model = true,
        .calibrated_model = mtpNaxCalibratedModelFrom(&model, 248320),
        .profiled_affine_trunk = true,
        .model_quant = .{ .bits = 4, .group_size = 64, .mode = .affine },
        .weight_present = true,
        .packed_weight = true,
        .scales_present = true,
        .biases_present = true,
        .quant = .{ .bits = 4, .group_size = 64, .mode = .affine },
        .K = 5120,
        .N = 248320,
        .packed_k = 640,
        .verify_on = true,
        .lane_on = true,
        .available = true,
        .min_m = 8,
    };
    try testing.expect(mtpNaxProfileEnabledFrom(good));

    var bad = good;
    bad.dense_model = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.calibrated_model = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.profiled_affine_trunk = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.bits = 8;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.group_size = 16;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.model_quant.mode = .nvfp4;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.weight_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.packed_weight = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.scales_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.biases_present = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    bad = good;
    bad.quant.bits = 8;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.quant.group_size = 16;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.quant.mode = .nvfp4;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    bad = good;
    bad.packed_k = 639;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.K = 5184;
    bad.packed_k = 648;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.N = 248324;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.N = 480;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    // Both depth-8 verify widths must take NAX. A min-M of 9 only covers one.
    bad = good;
    bad.min_m = 9;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.verify_on = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.lane_on = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));
    bad = good;
    bad.available = false;
    try testing.expect(!mtpNaxProfileEnabledFrom(bad));

    // Lowering the takeover width retains M=8/M=9 coverage.
    bad = good;
    bad.min_m = 5;
    try testing.expect(mtpNaxProfileEnabledFrom(bad));
}

test "NAX availability probe: G17 prefix + macOS 26.2 floor + fallback rehearsal (pure parts)" {
    // Mirrors MTPLX's shipping nax_available() gate exactly (nax_verify.py).
    // Arch: case-insensitive prefix match on "applegpu_g17".
    try testing.expect(naxArchIsG17("applegpu_g17s"));
    try testing.expect(naxArchIsG17("AppleGPU_G17"));
    try testing.expect(naxArchIsG17("applegpu_g17"));
    try testing.expect(!naxArchIsG17("applegpu_g16")); // M4-class
    try testing.expect(!naxArchIsG17("applegpu_g13"));
    try testing.expect(!naxArchIsG17(""));
    try testing.expect(!naxArchIsG17("g17"));

    // macOS floor: >= 26.2 (where MetalPerformancePrimitives ships).
    try testing.expect(macosVersionAtLeast("26.2", 26, 2));
    try testing.expect(macosVersionAtLeast("26.4.1", 26, 2));
    try testing.expect(macosVersionAtLeast("27.0", 26, 2));
    try testing.expect(macosVersionAtLeast("27", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.1", 26, 2));
    try testing.expect(!macosVersionAtLeast("26", 26, 2));
    try testing.expect(!macosVersionAtLeast("25.5", 26, 2));
    // Unparseable components read as 0 and never satisfy the floor (MTPLX
    // semantics: int() failures fall back to 0).
    try testing.expect(!macosVersionAtLeast("", 26, 2));
    try testing.expect(!macosVersionAtLeast("garbage", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.x", 26, 2));

    // The combined gate.
    try testing.expect(naxAvailableFrom(false, "applegpu_g17d", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g16", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g17d", "26.1"));
    // MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK=1 QA rehearsal: an M5 pretends the
    // units are absent so the exact M1-M4 plain-SIMD path runs there.
    try testing.expect(!naxAvailableFrom(true, "applegpu_g17d", "26.4"));
}

test "naxStatusFrom: on for G17 + 26.2, off names the missing leg (--version / Settings display)" {
    // Hardware + OS both satisfied → active. Kernels-present is a build-time
    // invariant for our binaries (build-mlx.sh + tests/test_mlx_staged_nax.sh),
    // so hardware capability == MLX's stock-op is_nax_available() gate.
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17s", "26.2"));
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17d", "27.0"));
    // Pre-M5 GPU: the GPU is the blocker regardless of OS.
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("applegpu_g16s", "26.4"));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("", "26.4"));
    // M5 on an old OS: the OS is the blocker (bundle min is 26.2 anyway, but
    // a dev build can run anywhere).
    try testing.expectEqualStrings("off (requires macOS 26.2+)", naxStatusFrom("applegpu_g17s", "26.1"));
}

test "verifyQmmNaxAvailable: false on every non-G17 device (kernel is never built here)" {
    // The real-device leg of the probe: on this M4 (and every CI runner) the
    // architecture is not applegpu_g17, so the probe must read false and the
    // NAX kernel object can never be constructed. On a real M5 the arch IS
    // G17 and this test only checks the plumbing returned something.
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse {
        // No Metal device info at all: the probe must be false.
        try testing.expect(!verifyQmmNaxAvailable());
        return;
    };
    try testing.expect(arch.len > 0);
    var vbuf: [64]u8 = undefined;
    const ver = macosProductVersion(&vbuf) orelse "<none>";
    std.debug.print("[nax-probe] arch={s} macos={s} available={}\n", .{ arch, ver, verifyQmmNaxAvailable() });
    if (!naxArchIsG17(arch)) {
        try testing.expect(!verifyQmmNaxAvailable());
    }
}

test "naxMinMFrom: default 8, clamped to [2,16], garbage ignored" {
    try testing.expectEqual(@as(c_int, 8), naxMinMFrom(null));
    try testing.expectEqual(@as(c_int, 5), naxMinMFrom("5"));
    try testing.expectEqual(@as(c_int, 16), naxMinMFrom("16"));
    try testing.expectEqual(@as(c_int, 2), naxMinMFrom("1"));
    try testing.expectEqual(@as(c_int, 2), naxMinMFrom("0"));
    try testing.expectEqual(@as(c_int, 16), naxMinMFrom("99"));
    try testing.expectEqual(@as(c_int, 8), naxMinMFrom("abc"));
    try testing.expectEqual(@as(c_int, 8), naxMinMFrom(""));
}

test "NAX host scaffolding: zero-pad to 16 rows + slice-back are exact (runs off-M5)" {
    // The half of runVerifyQmmNax that CAN execute without G17 hardware —
    // the exact production helpers (naxPadTo16 / naxSliceRows) wrapped
    // around STOCK qmm standing in for the NAX kernel. Pins the mlx_pad
    // axis/value plumbing (pad rows are EXACT zeros in the activation
    // dtype, real rows byte-preserved), the slice bounds, and that zero
    // activation rows produce exactly-zero output rows (so on the M5 the
    // padded tile positions can never bleed into the sliced result).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA110C);
    const rnd = prng.random();
    const K: c_int = 512;
    const N: c_int = 640; // % 32 == 0, NAX-shaped
    const gs: u32 = 64;

    const wn: usize = @intCast(N * K);
    const wbuf = try allocator.alloc(f32, wn);
    defer allocator.free(wbuf);
    for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
    const wshape = [_]c_int{ N, K };
    const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    for ([_]c_int{ 9, 16 }) |m| {
        const xn: usize = @intCast(m * K);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, m, K };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        // Pad half: [1, m, K] → [16, K]; real rows byte-preserved, pad rows 0.
        const x16 = try naxPadTo16(s, x, m, K, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        const psh = mlx.getShape(x16);
        try testing.expectEqual(@as(usize, 2), psh.len);
        try testing.expectEqual(@as(c_int, 16), psh[0]);
        try testing.expectEqual(K, psh[1]);
        {
            var xf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xf);
            var pf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pf);
            try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
            try mlx.check(mlx.mlx_astype(&pf, x16, .float32, s));
            try mlx.check(mlx.mlx_array_eval(xf));
            try mlx.check(mlx.mlx_array_eval(pf));
            const xd_ = mlx.mlx_array_data_float32(xf).?;
            const pd_ = mlx.mlx_array_data_float32(pf).?;
            for (0..xn) |i| try testing.expectEqual(xd_[i], pd_[i]);
            for (xn..@intCast(16 * K)) |i| try testing.expectEqual(@as(f32, 0.0), pd_[i]);
        }

        // Slice half around stock qmm: zero rows in → exactly-zero rows out,
        // and the slice returns the first m rows byte-identically.
        var y16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y16);
        try mlx.check(mlx.mlx_quantized_matmul(&y16, x16, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", s));
        const ym = try naxSliceRows(s, y16, m, N);
        defer _ = mlx.mlx_array_free(ym);
        const ysh = mlx.getShape(ym);
        try testing.expectEqual(@as(usize, 2), ysh.len);
        try testing.expectEqual(m, ysh[0]);
        try testing.expectEqual(N, ysh[1]);
        {
            var ff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff);
            var mf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mf);
            try mlx.check(mlx.mlx_astype(&ff, y16, .float32, s));
            try mlx.check(mlx.mlx_astype(&mf, ym, .float32, s));
            try mlx.check(mlx.mlx_array_eval(ff));
            try mlx.check(mlx.mlx_array_eval(mf));
            const fd = mlx.mlx_array_data_float32(ff).?;
            const md = mlx.mlx_array_data_float32(mf).?;
            const mn: usize = @intCast(m * N);
            for (0..mn) |i| try testing.expectEqual(fd[i], md[i]);
            for (mn..@intCast(16 * N)) |i| try testing.expectEqual(@as(f32, 0.0), @abs(fd[i]));
        }
    }
}

test "MoE decode gather µbench (MLX_SERVE_MOE_GATHER_UBENCH=1)" {
    // Reproduces the Laguna decode expert-gather with OUR self-built MLX, to
    // compare against the pip-MLX Python sim (single gather ~133us, 47-layer
    // MoE ~7.5ms). If this is far slower, the gap is our MLX build/runtime;
    // if it matches, the gap is the live server context (stream/alloc/pipeline).
    if (std.c.getenv("MLX_SERVE_MOE_GATHER_UBENCH") == null) return error.SkipZigTest;
    const io_util = @import("io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    // Trace mode: run the gather and qmv sections in long sustained bursts so an
    // Instruments Metal System Trace shows a clean, comparable GPU timeline.
    const gather_trace = std.c.getenv("MLX_SERVE_MOE_GATHER_TRACE") != null;
    const E: c_int = 256;
    const N: c_int = 1024; // moe_intermediate (gate/up out)
    const K: c_int = 3072; // hidden
    const TOPK: c_int = 10;
    const GS = 64;
    const BITS = 2;

    // Build a quantized [E,N,K] gate/up bank and a [E,K,N] down bank.
    const makeBank = struct {
        fn go(a: std.mem.Allocator, e: c_int, out: c_int, in: c_int, str: mlx.mlx_stream, rndp: *std.Random.DefaultPrng) ![3]mlx.mlx_array {
            const cnt: usize = @intCast(e * out * in);
            const buf = try a.alloc(f32, cnt);
            defer a.free(buf);
            const rr = rndp.random();
            for (buf) |*v| v.* = rr.float(f32) - 0.5;
            const wsh = [_]c_int{ e, out, in };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &wsh, 3, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, str));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", .{}, str));
            var wq = mlx.mlx_array_new();
            var wsc = mlx.mlx_array_new();
            var wbi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
            for ([_]mlx.mlx_array{ wq, wsc, wbi }) |aa| try mlx.check(mlx.mlx_array_eval(aa));
            return .{ wq, wsc, wbi };
        }
    }.go;

    var prng = std.Random.DefaultPrng.init(0xF00D);
    const gate = try makeBank(allocator, E, N, K, s, &prng);
    const down = try makeBank(allocator, E, K, N, s, &prng);
    defer for (gate ++ down) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // x [1,1,3072], inds [1,1,10]
    const xbuf = try allocator.alloc(f32, @intCast(K));
    defer allocator.free(xbuf);
    for (xbuf) |*v| v.* = prng.random().float(f32) - 0.5;
    const xsh = [_]c_int{ 1, 1, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
    const idxvals = [_]i32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
    const idxsh = [_]c_int{ 1, 1, TOPK };
    const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 3, .int32);
    defer _ = mlx.mlx_array_free(inds);
    const no_idx = mlx.mlx_array{ .ctx = null };

    _ = no_idx;
    // One decode gather at our 5-D shape, optionally biasing x by `add` to
    // defeat common-subexpression dedup when many are built in one graph.
    const oneGather = struct {
        fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
            const sh5 = [_]c_int{ 1, 1, 1, 1, K };
            var xe = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xe);
            try mlx.check(mlx.mlx_reshape(&xe, xin, &sh5, 5, str));
            if (add != 0.0) {
                const sc = mlx.mlx_array_new_float(add);
                defer _ = mlx.mlx_array_free(sc);
                var xa = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&xa, xe, sc, str));
                _ = mlx.mlx_array_free(xe);
                xe = xa;
            }
            var out5 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&out5, xe, bank[0], bank[1], bank[2], .{ .ctx = null }, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
            return out5;
        }
    }.go;

    var sw = io_util.Stopwatch.init(tio);
    // (1) single gather, 400 iters (cached) — compare to pip-MLX Python 133us
    {
        var i: usize = 0;
        while (i < 20) : (i += 1) {
            const g = try oneGather(s, x, gate, inds, 0.0);
            try mlx.check(mlx.mlx_array_eval(g));
            _ = mlx.mlx_array_free(g);
        }
        sw.reset();
        const ITER = 400;
        i = 0;
        while (i < ITER) : (i += 1) {
            const g = try oneGather(s, x, gate, inds, 0.0);
            try mlx.check(mlx.mlx_array_eval(g));
            _ = mlx.mlx_array_free(g);
        }
        const us = @as(f64, @floatFromInt(sw.read())) / 1000.0 / @as(f64, @floatFromInt(ITER));
        std.debug.print("\n[moe-gather-ubench] single gather_qmm (our MLX): {d:.1} us/call\n", .{us});
    }
    // (2) 141 INDEPENDENT gathers (varied x → no CSE) in ONE eval — measures
    // how well our MLX overlaps them (pip-MLX 47-layer MoE sim was ~7.5ms).
    {
        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneGather(s, x, gate, inds, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneGather(s, x, gate, inds, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 independent gathers, one eval (our MLX): {d:.2} ms  ({d:.1} us/gather)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (3) 141 INDEPENDENT quantized_matmul (dense qmv, NOT gather) in one eval —
    // does plain qmv overlap in our build where gather does not? If yes, an
    // engine-level workaround (extract experts + qmv) can dodge the gather bug.
    {
        // one 2D quantized weight [N, K] for qmv (transpose=true → x@w^T)
        const wn2: usize = @intCast(N * K);
        const wbuf2 = try allocator.alloc(f32, wn2);
        defer allocator.free(wbuf2);
        for (wbuf2) |*v| v.* = prng.random().float(f32) - 0.5;
        const wsh2 = [_]c_int{ N, K };
        const w2_32 = mlx.mlx_array_new_data(wbuf2.ptr, &wsh2, 2, .float32);
        defer _ = mlx.mlx_array_free(w2_32);
        var w2b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w2b);
        try mlx.check(mlx.mlx_astype(&w2b, w2_32, .bfloat16, s));
        var trip = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(trip);
        try mlx.check(mlx.mlx_quantize(&trip, w2b, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", .{}, s));
        var q = mlx.mlx_array_new();
        var sc = mlx.mlx_array_new();
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        defer _ = mlx.mlx_array_free(sc);
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&q, trip, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, trip, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, trip, 2));
        for ([_]mlx.mlx_array{ q, sc, bi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        const oneQmv = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, wq_: mlx.mlx_array, wsc_: mlx.mlx_array, wbi_: mlx.mlx_array, add: f32) !mlx.mlx_array {
                var xe = mlx.mlx_array_new();
                if (add != 0.0) {
                    const scl = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(scl);
                    try mlx.check(mlx.mlx_add(&xe, xin, scl, str));
                } else {
                    try mlx.check(mlx.mlx_astype(&xe, xin, .bfloat16, str));
                }
                defer _ = mlx.mlx_array_free(xe);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&out, xe, wq_, wsc_, wbi_, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", str));
                return out;
            }
        }.go;

        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneQmv(s, x, q, sc, bi, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneQmv(s, x, q, sc, bi, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 independent quantized_matmul (qmv), one eval (our MLX): {d:.2} ms  ({d:.1} us/qmv)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (4) FULL WORKAROUND: take 10 experts from the [256,N,K] bank + broadcast x
    // + BATCHED quantized_matmul. Includes the extraction cost. If fast, this is
    // the decode drop-in for gather_qmm.
    {
        const idx10 = [_]i32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
        const idx10sh = [_]c_int{10};
        const inds10 = mlx.mlx_array_new_data(&idx10, &idx10sh, 1, .int32);
        defer _ = mlx.mlx_array_free(inds10);
        // take from the real 256-expert `gate` bank each call (extraction cost)
        const oneBatched = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
                var wq_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wq_);
                var wsc_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wsc_);
                var wbi_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wbi_);
                try mlx.check(mlx.mlx_take_axis(&wq_, bank[0], ind, 0, str));
                try mlx.check(mlx.mlx_take_axis(&wsc_, bank[1], ind, 0, str));
                try mlx.check(mlx.mlx_take_axis(&wbi_, bank[2], ind, 0, str));
                const sh3 = [_]c_int{ 1, 1, K };
                var xe = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&xe, xin, &sh3, 3, str));
                if (add != 0.0) {
                    const scl = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(scl);
                    var xa = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&xa, xe, scl, str));
                    _ = mlx.mlx_array_free(xe);
                    xe = xa;
                }
                defer _ = mlx.mlx_array_free(xe);
                const bsh = [_]c_int{ 10, 1, K };
                var xbc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xbc);
                try mlx.check(mlx.mlx_broadcast_to(&xbc, xe, &bsh, 3, str));
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&out, xbc, wq_, wsc_, wbi_, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", str));
                return out; // [10,1,N]
            }
        }.go;

        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneBatched(s, x, gate, inds10, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneBatched(s, x, gate, inds10, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 BATCHED qmv over 10 experts, one eval (our MLX): {d:.2} ms  ({d:.1} us/batched)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (4b) OUR gatherQmv kernel: reads the bank IN PLACE with GPU-resident
    // indices, so it moves the ideal 9.8 MB instead of the batched path's 3x.
    // Target is pip-Python's gather number (~22 us), not the batched 70 us.
    {
        const idx10 = [_]u32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
        const idx10sh = [_]c_int{TOPK};
        const inds_u32 = mlx.mlx_array_new_data(&idx10, &idx10sh, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds_u32);
        const xk = [_]c_int{K};
        var xflat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xflat);
        try mlx.check(mlx.mlx_reshape(&xflat, x, &xk, 1, s));
        try mlx.check(mlx.mlx_array_eval(xflat));

        const NG = 141;
        const ITER: usize = if (gather_trace) 200 else 20;
        var pass: usize = 0;
        while (pass < ITER + 1) : (pass += 1) {
            if (pass == 1) sw.reset();
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = (try gatherQmv(s, xflat, gate[0], gate[1], gate[2], inds_u32, BITS, GS, .affine, false)).?;
                defer _ = mlx.mlx_array_free(g);
                try mlx.check(mlx.mlx_vector_array_append_value(vec, g));
            }
            try mlx.check(mlx.mlx_eval(vec));
        }
        const gms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 OUR gatherQmv (in-place bank), one eval: {d:.2} ms  ({d:.1} us/gather)\n", .{ gms, gms * 1000.0 / @as(f64, NG) });
    }
    // (5) ISOLATION MATRIX. A solo gather is ~144 us but 141-in-one-graph is
    // ~349 us EACH — batching makes each gather SLOWER than running it alone,
    // so this is not merely "fails to overlap". Two candidate serializers, each
    // testable independently (2x2):
    //   - add-chain: the accumulate in (2) is a serial dependency chain, and
    //     every dependent op forces a FULL `memoryBarrier(BarrierScopeBuffers)`
    //     in MLX's concurrent command encoder (backend/metal/device.cpp
    //     maybeInsertBarrier) — one barrier per link drains the whole GPU.
    //   - implicit lhs_indices: `gather_qmm` with a null lhs runs
    //     `indices_or_default` (ops.cpp), which emits a per-call
    //     arange+reshape. The gather then READS that fresh arange output, so
    //     each gather is data-dependent on a kernel dispatched immediately
    //     before it => another guaranteed barrier per gather. Plain qmv has no
    //     index prep at all, which is exactly why it overlaps.
    // Passing an explicit lhs (broadcastable zeros) hoists the index array out
    // of the per-call graph and should remove the second barrier entirely.
    {
        const NG = 141;
        // Explicit lhs_indices, built ONCE and reused by every gather: uint32
        // zeros shaped [1,1,TOPK] so it broadcasts against rhs without work.
        const lhs_vals = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // TOPK zeros
        const lhs_sh = [_]c_int{ 1, 1, TOPK };
        const lhs_idx = mlx.mlx_array_new_data(&lhs_vals, &lhs_sh, 3, .uint32);
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_array_eval(lhs_idx));

        const oneGatherLhs = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
                const sh5 = [_]c_int{ 1, 1, 1, 1, K };
                var xe = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xe);
                try mlx.check(mlx.mlx_reshape(&xe, xin, &sh5, 5, str));
                if (add != 0.0) {
                    const sc = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(sc);
                    var xa = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&xa, xe, sc, str));
                    _ = mlx.mlx_array_free(xe);
                    xe = xa;
                }
                var out5 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_gather_qmm(&out5, xe, bank[0], bank[1], bank[2], lhs, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
                return out5;
            }
        }.go;

        // Run one cell of the matrix: `explicit_lhs` x `chain` (add-chain vs a
        // flat vector eval of 141 independent results).
        const cell = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, explicit_lhs: bool, chain: bool, iters: usize, watch: *io_util.Stopwatch) !f64 {
                var pass: usize = 0;
                var total_ns: u64 = 0;
                // one warmup pass then `iters` timed passes
                while (pass < iters + 1) : (pass += 1) {
                    if (pass == 1) watch.reset();
                    var acc = mlx.mlx_array_new();
                    var first = true;
                    const vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(vec);
                    var j: usize = 0;
                    while (j < NG) : (j += 1) {
                        const bias = @as(f32, @floatFromInt(j)) * 0.001;
                        const g = if (explicit_lhs)
                            try oneGatherLhs(str, xin, bank, lhs, ind, bias)
                        else
                            try oneGather(str, xin, bank, ind, bias);
                        if (!chain) {
                            try mlx.check(mlx.mlx_vector_array_append_value(vec, g));
                            _ = mlx.mlx_array_free(g);
                            continue;
                        }
                        if (first) {
                            acc = g;
                            first = false;
                        } else {
                            var na = mlx.mlx_array_new();
                            try mlx.check(mlx.mlx_add(&na, acc, g, str));
                            _ = mlx.mlx_array_free(acc);
                            _ = mlx.mlx_array_free(g);
                            acc = na;
                        }
                    }
                    if (chain) {
                        try mlx.check(mlx.mlx_array_eval(acc));
                        _ = mlx.mlx_array_free(acc);
                    } else {
                        try mlx.check(mlx.mlx_eval(vec));
                    }
                }
                total_ns = watch.read();
                return @as(f64, @floatFromInt(total_ns)) / 1.0e6 / @as(f64, @floatFromInt(iters));
            }
        }.go;

        const ITER: usize = if (gather_trace) 200 else 20;
        std.debug.print("[moe-gather-ubench] --- isolation matrix (141 gathers, one eval) ---\n", .{});
        for ([_]bool{ false, true }) |explicit_lhs| {
            for ([_]bool{ true, false }) |chain| {
                const ms = try cell(s, x, gate, lhs_idx, inds, explicit_lhs, chain, ITER, &sw);
                std.debug.print("[moe-gather-ubench]   lhs={s:<8} accum={s:<9} {d:6.2} ms  ({d:.1} us/gather)\n", .{
                    if (explicit_lhs) "explicit" else "implicit",
                    if (chain) "add-chain" else "none",
                    ms,
                    ms * 1000.0 / @as(f64, NG),
                });
            }
        }

        // (6) FULLY INDEPENDENT: every input pre-materialized, so the 141
        // gathers have NO producer op in the graph at all — no add feeding x,
        // no arange feeding the indices, no accumulate consuming the outputs.
        // If these still fail to overlap, no data dependency (and therefore no
        // MLX barrier) is responsible and the cost is the kernel itself.
        // The qmv control runs on the SAME pre-materialized x list.
        {
            var xs: [NG]mlx.mlx_array = undefined;
            for (&xs, 0..) |*slot, j| {
                const sh5 = [_]c_int{ 1, 1, 1, 1, K };
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_reshape(&xr, x, &sh5, 5, s));
                const scl = mlx.mlx_array_new_float(@as(f32, @floatFromInt(j)) * 0.001);
                defer _ = mlx.mlx_array_free(scl);
                var xa = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&xa, xr, scl, s));
                try mlx.check(mlx.mlx_array_eval(xa)); // materialize: no producer left
                slot.* = xa;
            }
            defer for (xs) |a| {
                _ = mlx.mlx_array_free(a);
            };

            const indep = struct {
                fn go(str: mlx.mlx_stream, xlist: []const mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, iters: usize, watch: *io_util.Stopwatch) !f64 {
                    var pass: usize = 0;
                    while (pass < iters + 1) : (pass += 1) {
                        if (pass == 1) watch.reset();
                        const vec = mlx.mlx_vector_array_new();
                        defer _ = mlx.mlx_vector_array_free(vec);
                        for (xlist) |xj| {
                            var out5 = mlx.mlx_array_new();
                            defer _ = mlx.mlx_array_free(out5);
                            try mlx.check(mlx.mlx_gather_qmm(&out5, xj, bank[0], bank[1], bank[2], lhs, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
                            try mlx.check(mlx.mlx_vector_array_append_value(vec, out5));
                        }
                        try mlx.check(mlx.mlx_eval(vec));
                    }
                    return @as(f64, @floatFromInt(watch.read())) / 1.0e6 / @as(f64, @floatFromInt(iters));
                }
            }.go;

            const ms_impl = try indep(s, &xs, gate, .{ .ctx = null }, inds, ITER, &sw);
            std.debug.print("[moe-gather-ubench]   lhs=implicit inputs=premat {d:6.2} ms  ({d:.1} us/gather)\n", .{ ms_impl, ms_impl * 1000.0 / @as(f64, NG) });
            const ms_expl = try indep(s, &xs, gate, lhs_idx, inds, ITER, &sw);
            std.debug.print("[moe-gather-ubench]   lhs=explicit inputs=premat {d:6.2} ms  ({d:.1} us/gather)\n", .{ ms_expl, ms_expl * 1000.0 / @as(f64, NG) });

            // (7) BANK-SIZE SWEEP. Every cell below reads the SAME 10 experts
            // (identical bytes touched, identical FLOPs) — only the size of the
            // bank they are indexed out of changes. A flat curve means the
            // gather kernel is uniformly slow; a curve that rises with E means
            // the cost is bank-extent addressing (TLB/cache), not the math.
            const E_SWEEP = [_]c_int{ 10, 32, 64, 256 };
            // MLX's default wired limit is 0 (backend/metal/allocator.cpp), so
            // the residency set wires NOTHING and every expert bank lives in
            // `unwired_set_`. Run the sweep at the default and again with the
            // banks wired: if the O(E) slope collapses when wired, the cost is
            // per-dispatch residency/page validation of the bound allocation,
            // not the gather math.
            // Python's sweep runs from a clean allocator (active=0/cache=0) and
            // is FLAT in E; ours runs after ~11 GB of buffer cache has piled up
            // from building the banks. Sweep both ways: `clear` drops MLX's
            // cached MTLBuffers first.
            for ([_]bool{ false, true }) |clear_first| {
                if (clear_first) try mlx.check(mlx.mlx_clear_cache());
                var prev: usize = 0;
                const wired: usize = 0;
                try mlx.check(mlx.mlx_set_wired_limit(&prev, wired));
                var a0: usize = 0;
                var c0: usize = 0;
                try mlx.check(mlx.mlx_get_active_memory(&a0));
                try mlx.check(mlx.mlx_get_cache_memory(&c0));
                std.debug.print("[moe-gather-ubench]   -- clear_cache_first={} (active={d} MB cache={d} MB) --\n", .{ clear_first, a0 >> 20, c0 >> 20 });
                for (E_SWEEP) |e| {
                    var sweep_prng = std.Random.DefaultPrng.init(0xC0FFEE);
                    const bank = try makeBank(allocator, e, N, K, s, &sweep_prng);
                    defer for (bank) |a| {
                        _ = mlx.mlx_array_free(a);
                    };
                    // Re-apply the limit so banks allocated just now get pulled
                    // out of unwired_set_ by ResidencySet::resize.
                    var p2: usize = 0;
                    try mlx.check(mlx.mlx_set_wired_limit(&p2, 0));
                    try mlx.check(mlx.mlx_set_wired_limit(&p2, wired));
                    // indices must stay in range for the small banks; keep the same
                    // COUNT (10) so the work per gather is identical across cells.
                    var iv: [10]i32 = undefined;
                    for (&iv, 0..) |*slot, j| slot.* = @intCast(@mod(@as(c_int, @intCast(j)), e));
                    const isz = [_]c_int{ 1, 1, TOPK };
                    const ind_e = mlx.mlx_array_new_data(&iv, &isz, 3, .int32);
                    defer _ = mlx.mlx_array_free(ind_e);
                    const ms_e = try indep(s, &xs, bank, lhs_idx, ind_e, ITER, &sw);
                    const mb = @as(f64, @floatFromInt(@as(i64, e) * N * K)) * 0.25 / 1.0e6;
                    std.debug.print("[moe-gather-ubench]   E={d:<4} bank={d:5.0} MB   {d:6.2} ms  ({d:.1} us/gather)\n", .{ e, mb, ms_e, ms_e * 1000.0 / @as(f64, NG) });
                }
            }
            var restore: usize = 0;
            try mlx.check(mlx.mlx_set_wired_limit(&restore, 0));
            var act: usize = 0;
            var cch: usize = 0;
            var pk: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&act));
            try mlx.check(mlx.mlx_get_cache_memory(&cch));
            try mlx.check(mlx.mlx_get_peak_memory(&pk));
            std.debug.print("[moe-gather-ubench]   mem: active={d} MB cache={d} MB peak={d} MB\n", .{ act >> 20, cch >> 20, pk >> 20 });
        }
    }
}

test "verifyQmm µbench: kernel vs stock per 27B shape (MLX_SERVE_VQMM_UBENCH=1)" {
    if (std.c.getenv("MLX_SERVE_VQMM_UBENCH") == null) return error.SkipZigTest;
    const io_util = @import("io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rnd = prng.random();
    const WARM = 5;
    const ITERS = 30;

    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int }{
        .{ .name = "qkvz", .k = 5120, .n = 16384 },
        .{ .name = "gate/up", .k = 5120, .n = 17408 },
        .{ .name = "down", .k = 17408, .n = 5120 },
        .{ .name = "out", .k = 6144, .n = 5120 },
        .{ .name = "lm_head", .k = 5120, .n = 151936 },
    };
    std.debug.print("\n[vqmm-ubench] {s:>8} {s:>3} {s:>10} {s:>10} {s:>8}\n", .{ "shape", "M", "stock_ms", "kernel_ms", "ratio" });
    for (shapes) |sh| {
        const wn: usize = @intCast(sh.n * sh.k);
        const wbuf = try allocator.alloc(f32, wn);
        for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const wshape = [_]c_int{ sh.n, sh.k };
        const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
        allocator.free(wbuf);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
        for ([_]mlx.mlx_array{ wq, wsc, wbi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        // M 4/6 exercise the split-K lanes; 8/12/16 exercise the NAX m16
        // tile on M5-class machines and honestly print "fallback" elsewhere.
        // For the M5-day NAX-at-low-M A/B (todo-m5-nax.md §7), re-run with
        // MLX_SERVE_VERIFY_QMM_NAX_MIN_M=4 so the 4/6 rows route to NAX.
        for ([_]c_int{ 4, 6, 8, 12, 16 }) |m| {
            const xn: usize = @intCast(m * sh.k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, sh.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try mlx.check(mlx.mlx_array_eval(x));

            var stock_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_quantized_matmul(&out, x, wq, wsc, wbi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
                    try mlx.check(mlx.mlx_array_eval(out));
                    _ = mlx.mlx_array_free(out);
                }
                stock_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            var kern_ms: f64 = 0;
            var engaged = true;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    const out_opt = try verifyQmm(s, x, wq, wsc, wbi, 4, 64);
                    const out = out_opt orelse {
                        engaged = false;
                        break;
                    };
                    try mlx.check(mlx.mlx_array_eval(out));
                    _ = mlx.mlx_array_free(out);
                }
                kern_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            if (engaged) {
                std.debug.print("[vqmm-ubench] {s:>8} {d:>3} {d:>10.3} {d:>10.3} {d:>8.2}\n", .{ sh.name, m, stock_ms, kern_ms, kern_ms / stock_ms });
            } else {
                std.debug.print("[vqmm-ubench] {s:>8} {d:>3} {d:>10.3} {s:>10} {s:>8}\n", .{ sh.name, m, stock_ms, "fallback", "-" });
            }
        }
    }
}

test "verifyQmm mixed-width NAX µbench (MLX_SERVE_VQMM_MIXED_UBENCH=1)" {
    if (std.c.getenv("MLX_SERVE_VQMM_MIXED_UBENCH") == null) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const io_util = @import("io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0C4E);
    const rnd = prng.random();
    const WARM = 10;
    const ITERS = 80;
    const M: c_int = 8;

    const shapes = [_]struct { name: []const u8, bits: u32, k: c_int, n: c_int }{
        .{ .name = "q4-gate", .bits = 4, .k = 5120, .n = 17408 },
        .{ .name = "q5-in_z", .bits = 5, .k = 5120, .n = 6144 },
        .{ .name = "q5-qproj", .bits = 5, .k = 5120, .n = 12288 },
        .{ .name = "q5-kproj", .bits = 5, .k = 5120, .n = 1024 },
        .{ .name = "q5-out", .bits = 5, .k = 6144, .n = 5120 },
        .{ .name = "q5-down", .bits = 5, .k = 17408, .n = 5120 },
        .{ .name = "q6-v", .bits = 6, .k = 5120, .n = 1024 },
        .{ .name = "q6-down", .bits = 6, .k = 17408, .n = 5120 },
    };
    std.debug.print("\n[vqmm-mixed] {s:>8} {s:>4} {s:>10} {s:>10} {s:>8}\n", .{
        "shape", "bits", "stock_ms", "nax_ms", "ratio",
    });
    for (shapes) |sh| {
        const wn: usize = @intCast(sh.n * sh.k);
        const wbuf = try allocator.alloc(f32, wn);
        for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const wshape = [_]c_int{ sh.n, sh.k };
        const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
        allocator.free(wbuf);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            wb,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(@intCast(sh.bits)),
            "affine",
            .{},
            s,
        ));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
        for ([_]mlx.mlx_array{ wq, wsc, wbi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        const xn: usize = @intCast(M * sh.k);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, M, sh.k };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(x));

        // Interleave the two paths (and reverse their order every iteration)
        // so GPU frequency/thermal drift cannot systematically favor the path
        // measured second.
        var it: usize = 0;
        while (it < WARM) : (it += 1) {
            var stock = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(
                &stock,
                x,
                wq,
                wsc,
                wbi,
                true,
                mlx.mlx_optional_int.some(64),
                mlx.mlx_optional_int.some(@intCast(sh.bits)),
                "affine",
                s,
            ));
            try mlx.check(mlx.mlx_array_eval(stock));
            _ = mlx.mlx_array_free(stock);
            const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                return error.TestUnexpectedResult;
            try mlx.check(mlx.mlx_array_eval(nax));
            _ = mlx.mlx_array_free(nax);
        }
        var stock_ns: u64 = 0;
        var nax_ns: u64 = 0;
        it = 0;
        while (it < ITERS) : (it += 1) {
            if (it % 2 == 0) {
                var sw = io_util.Stopwatch.init(tio);
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(
                    &stock,
                    x,
                    wq,
                    wsc,
                    wbi,
                    true,
                    mlx.mlx_optional_int.some(64),
                    mlx.mlx_optional_int.some(@intCast(sh.bits)),
                    "affine",
                    s,
                ));
                try mlx.check(mlx.mlx_array_eval(stock));
                _ = mlx.mlx_array_free(stock);
                stock_ns += sw.read();
                sw.reset();
                const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                    return error.TestUnexpectedResult;
                try mlx.check(mlx.mlx_array_eval(nax));
                _ = mlx.mlx_array_free(nax);
                nax_ns += sw.read();
            } else {
                var sw = io_util.Stopwatch.init(tio);
                const nax = (try verifyQmm(s, x, wq, wsc, wbi, sh.bits, 64)) orelse
                    return error.TestUnexpectedResult;
                try mlx.check(mlx.mlx_array_eval(nax));
                _ = mlx.mlx_array_free(nax);
                nax_ns += sw.read();
                sw.reset();
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(
                    &stock,
                    x,
                    wq,
                    wsc,
                    wbi,
                    true,
                    mlx.mlx_optional_int.some(64),
                    mlx.mlx_optional_int.some(@intCast(sh.bits)),
                    "affine",
                    s,
                ));
                try mlx.check(mlx.mlx_array_eval(stock));
                _ = mlx.mlx_array_free(stock);
                stock_ns += sw.read();
            }
        }
        const stock_ms = @as(f64, @floatFromInt(stock_ns)) / @as(f64, ITERS) / 1e6;
        const nax_ms = @as(f64, @floatFromInt(nax_ns)) / @as(f64, ITERS) / 1e6;
        std.debug.print("[vqmm-mixed] {s:>8} {d:>4} {d:>10.3} {d:>10.3} {d:>8.2}\n", .{
            sh.name, sh.bits, stock_ms, nax_ms, nax_ms / stock_ms,
        });
    }
}

test "prefill qmm µbench: stock qmm vs dequant+GEMM at 27B prefill shapes (MLX_SERVE_PREFILL_QMM_UBENCH=1)" {
    if (std.c.getenv("MLX_SERVE_PREFILL_QMM_UBENCH") == null) return error.SkipZigTest;
    const io_util = @import("io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xFEED);
    const rnd = prng.random();
    const WARM = 3;
    const ITERS = 10;

    // The 27B oQ4e prefill shapes: MLP gate/up (q4 gs64), MLP down (q5
    // gs64), GDN qkvz (q4). oMLX's prefill patch replaces exactly this class
    // with retiled qmm_t (bm 64-128 vs stock 32) — measure whether
    // dequant+dense-GEMM (steel bf16 gemm at near-peak) beats stock qmm
    // per-call, and what the amortized (pre-dequantized) ceiling is.
    const shapes = [_]struct { name: []const u8, k: c_int, n: c_int, bits: c_int }{
        .{ .name = "gate/up-q4", .k = 5120, .n = 17408, .bits = 4 },
        .{ .name = "down-q5", .k = 17408, .n = 5120, .bits = 5 },
        .{ .name = "qkvz-q4", .k = 5120, .n = 16384, .bits = 4 },
    };
    std.debug.print("\n[pqmm-ubench] {s:>10} {s:>5} {s:>9} {s:>12} {s:>12}\n", .{ "shape", "M", "stock_ms", "dq+gemm_ms", "gemm_ms" });
    for (shapes) |sh| {
        const wn: usize = @intCast(sh.n * sh.k);
        const wbuf = try allocator.alloc(f32, wn);
        for (wbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const wshape = [_]c_int{ sh.n, sh.k };
        const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
        allocator.free(wbuf);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
        for ([_]mlx.mlx_array{ wq, wsc, wbi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        // Pre-dequantized weight for the amortized-GEMM ceiling.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .bfloat16, .has_value = true }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        try mlx.check(mlx.mlx_transpose(&wdq_t, wdq, s));
        try mlx.check(mlx.mlx_array_eval(wdq_t));

        for ([_]c_int{ 2048, 4096 }) |m| {
            const xn: usize = @intCast(m * sh.k);
            const xbuf = try allocator.alloc(f32, xn);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, sh.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            allocator.free(xbuf);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try mlx.check(mlx.mlx_array_eval(x));

            var stock_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    const r1 = try qmatmulBits(x, wq, wsc, wbi, @intCast(sh.bits), 64, .affine, s);
                    try mlx.check(mlx.mlx_array_eval(r1));
                    _ = mlx.mlx_array_free(r1);
                }
                stock_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            var dq_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var dq = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_dequantize(&dq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(sh.bits), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .bfloat16, .has_value = true }, s));
                    var dq_t = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
                    var r2 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&r2, x, dq_t, s));
                    try mlx.check(mlx.mlx_array_eval(r2));
                    _ = mlx.mlx_array_free(dq);
                    _ = mlx.mlx_array_free(dq_t);
                    _ = mlx.mlx_array_free(r2);
                }
                dq_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            var gemm_ms: f64 = 0;
            {
                var it: usize = 0;
                var sw = io_util.Stopwatch.init(tio);
                while (it < WARM + ITERS) : (it += 1) {
                    if (it == WARM) sw.reset();
                    var r3 = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_matmul(&r3, x, wdq_t, s));
                    try mlx.check(mlx.mlx_array_eval(r3));
                    _ = mlx.mlx_array_free(r3);
                }
                gemm_ms = @as(f64, @floatFromInt(sw.read())) / @as(f64, ITERS) / 1e6;
            }
            _ = mlx.mlx_clear_cache();
            std.debug.print("[pqmm-ubench] {s:>10} {d:>5} {d:>9.2} {d:>12.2} {d:>12.2}\n", .{ sh.name, m, stock_ms, dq_ms, gemm_ms });
        }
    }
}

test "GDN µbench: sequential kernel vs bare qmm at 27B shapes (attribution; MLX_SERVE_GDN_UBENCH=1)" {
    // ATTRIBUTION probe, not a pass/fail guard (live A/Bs decide shipping):
    // decomposes the multi-token forward ladder into (a) the GDN recurrence
    // kernel's sequential-over-T cost and (b) qmm row-count effects, at the
    // real Qwen3.6-27B geometry. Run with:
    //   MLX_SERVE_GDN_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="GDN µbench"
    if (std.c.getenv("MLX_SERVE_GDN_UBENCH") == null) return error.SkipZigTest;
    const io_util = @import("io_util.zig");
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
                const out_state = try gdnTestRun(false, q, k, v, g, beta, st0, B, T, Hk, Hv, Dk, Dv, s);
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

test "prefillEvalCadence: small transients keep the coarse cadence" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 26B geometry, short prompt: 16 heads x 2048 chunk x 2048 kv x 2B = 128 MB
    // of unfused scores — well inside the budget, keep today's cadence.
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 2048, 2048, false));
    // The LIVE-measured no-regression point: a 5,140-token agent prompt
    // (845 MB scores) must keep cadence 4 — flipping it cost a measured 4.5%
    // prefill for zero memory benefit (peak 18.1 GB both ways).
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 5140, 5140, false));
    // Fused head dim + dense KV: no transient at all, keep even the 48 default.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 500_000, false));
}

test "prefillEvalCadence: fused hd-256 kernel drops the score term" {
    const t = std.testing;
    // With msv_attn_p256 active (the default) the score transient never
    // exists — hd 256 keeps the coarse cadence at any context, exactly like
    // hd 128. The dequant term must still fire under --kv-quant.
    fused256_override = true;
    defer fused256_override = null;
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 102_448, false));
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 400_000, true));
}

test "prefillEvalCadence: big unfused score tensor forces eval-per-layer" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 16 heads x 8192 chunk x 16384 kv x 2B = 4 GiB > 2 GiB budget: bounds a
    // ~13 GB windowed transient to ~4 GB. (8192 chunk x 8192 kv sits exactly
    // ON the 2 GiB budget and deliberately keeps the coarse cadence.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 16384, false));
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 8192, false));
    // The capped 1024 chunk at 102K ctx still materializes 3.4 GB — the chunk
    // cap (generate.zig) does NOT restore the coarse cadence at long context.
    // (Live: this cadence measured peak 27.0 GB and +14% prefill vs baseline.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 1024, 102_448, false));
}

test "prefillEvalCadenceApplies: spec-verify-width forwards skip the cadence entirely" {
    const t = std.testing;
    // Spec-decode verify forwards (PLD/drafter/MTP: seq 2..9) are
    // decode-shaped — KB-scale transients — but ride the prefill layer loop
    // because they're multi-token. The mid-loop eval() cadence only costs
    // them synchronous pipeline drains (measured: cadence 4 on the 64-layer
    // qwen3.6-27B = 16 drains per MTP round, ~13 ms of a 48 ms depth-1
    // round, superlinear in draft depth). seq-1 decode never ran cadence
    // evals, so exempting verify widths bounds nothing new.
    try t.expect(!Transformer.prefillEvalCadenceApplies(2));
    try t.expect(!Transformer.prefillEvalCadenceApplies(9));
    try t.expect(!Transformer.prefillEvalCadenceApplies(31));
    // Real prefill chunks keep the cadence (and the budget-driven flip).
    try t.expect(Transformer.prefillEvalCadenceApplies(32));
    try t.expect(Transformer.prefillEvalCadenceApplies(512));
    try t.expect(Transformer.prefillEvalCadenceApplies(8192));
}

test "prefillEvalCadence: quantized-KV dequant forces eval-per-layer even at fused head dims" {
    const t = std.testing;
    // hd 128 is fused (zero scores) but denseView rebuilds the FULL cache per
    // layer under --kv-quant: 2 x 600K x 8 x 128 x 2B = 2.46 GB > 2 GiB.
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, true));
    // Same shape with dense fp16 KV: nothing to dequantize, keep coarse.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, false));
}

// ── fused hd-256 prefill attention parity ──

/// Composed reference for the fused kernel tests: MLX's own SDPA (which takes
/// the composed matmul→softmax→matmul path at head_dim 256).
fn attn256Reference(
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
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, mode, if (mask.ctx != null) mask else none, .{ .ctx = null }, s));
    return ref;
}

fn attn256RandBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
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

fn attn256MaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, s));
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

test "fusedSdpa256Prefill: causal parity vs composed SDPA (GQA, ragged shapes, chunk offset)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x256256);
    const rnd = prng.random();

    // qL=70 (partial 32-tile), kL=193 (partial 16-block), offset 123 (chunked
    // prefill bottom-right alignment), Hq=6/Hk=2 (gqa 3).
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0; // 1/sqrt(256)

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    // Not byte-parity: the composed path rounds softmax probs to bf16 before
    // the AV matmul, the fused kernel keeps float32 — bf16-rounding-scale
    // differences only.
    // Measured 0.00049 (one bf16 ULP at 0.5 magnitude) on this seed.
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

test "fusedSdpa256Prefill: sliding-band parity vs composed 'array' mask (Gemma local layers)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0;

    // Reference mask: exactly createSlidingWindowMask semantics — masked when
    // col > row_abs (causal) or row_abs - col >= sw, row_abs = kL - qL + r.
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpa256Prefill: declines cleanly outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();

    // Wrong head_dim -> null.
    const q128_shape = [_]c_int{ 1, 4, 64, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    fused256_override = true;
    defer fused256_override = null;
    try std.testing.expect((try fusedSdpa256Prefill(s, q128, q128, q128, 1.0, 0)) == null);

    // Decode/verify shapes (q_len < 16) -> null (sdpa_vector owns hd 256
    // there; stealing MTP verify forwards measured decode 48 -> 18 tok/s;
    // the 16 floor matches oMLX's _MIN_ROUTE_Q_LEN — with causal fused
    // default-on, a depth-8 MTP verify is q_len 9 and must NOT route here).
    const q1_shape = [_]c_int{ 1, 4, 1, 256 };
    const q8_shape = [_]c_int{ 1, 4, 8, 256 };
    const q15_shape = [_]c_int{ 1, 4, 15, 256 };
    const kv_shape = [_]c_int{ 1, 4, 64, 256 };
    const q1 = try attn256RandBf16(rnd, &q1_shape, s);
    defer _ = mlx.mlx_array_free(q1);
    const q8 = try attn256RandBf16(rnd, &q8_shape, s);
    defer _ = mlx.mlx_array_free(q8);
    const q15 = try attn256RandBf16(rnd, &q15_shape, s);
    defer _ = mlx.mlx_array_free(q15);
    const k1 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k1);
    try std.testing.expect((try fusedSdpa256Prefill(s, q1, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q8, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpa256Prefill(s, q15, k1, k1, 1.0, 0)) == null);

    // Kill switch -> null even for a conforming call.
    fused256_override = false;
    const q_shape = [_]c_int{ 1, 4, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    try std.testing.expect((try fusedSdpa256Prefill(s, q, q, q, 1.0, 0)) == null);
}

test "fusedSdpa256Prefill: causal and band both default FUSED (budgeted-dispatch flip)" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4A7E);
    const rnd = prng.random();

    // No override, no env: BOTH arms engage. The causal arm's historical
    // net-loss (every pre-budget ratio-gated variant lost same-boot on the
    // 27B) was the IOGPU preemption class — with the kv-chunk dispatch
    // budget it wins live (2026-07-22 same-session A/B: +2.9%/+2.3%/+4.6%
    // at 8K/16K/32K on the 27B), so causal is default-on now.
    // MLX_SERVE_FUSED_256_CAUSAL=0 restores composed causal.
    std.debug.assert(fused256_override == null);
    const q_shape = [_]c_int{ 1, 6, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    const causal = try fusedSdpa256Prefill(s, q, k, k, 1.0, 0);
    try std.testing.expect(causal != null);
    if (causal) |f| _ = mlx.mlx_array_free(f);

    const banded = try fusedSdpa256Prefill(s, q, k, k, 1.0, 40);
    try std.testing.expect(banded != null);
    if (banded) |f| _ = mlx.mlx_array_free(f);
}

test "fused256KvChunkLen: BK alignment, one-block floor, kl cap, budget-off" {
    const t = std.testing;
    // budget <= 0: single dispatch covering the full key axis.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 0));
    try t.expectEqual(@as(c_int, 65536), fused256KvChunkLen(1, 24, 2048, 65536, -1));
    // 27B live shape (24 heads, 2048-chunk q) at the default budget:
    // 250M / (24*2048) = 5086 keys -> floored to the BK(32) multiple 5056.
    try t.expectEqual(@as(c_int, 5056), fused256KvChunkLen(1, 24, 2048, 65536, FUSED256_DEFAULT_DISPATCH_BUDGET));
    // Never below one BK block even at absurdly small budgets…
    try t.expectEqual(@as(c_int, 32), fused256KvChunkLen(1, 24, 2048, 65536, 1));
    // …and never above the actual key length.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 1 << 40));
}

test "fusedSdpa256Prefill: budgeted kv chunking engages and is exact vs single dispatch" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xC4A2);
    const rnd = prng.random();

    // Hq=6/Hk=2, qL=70 (ragged q tile), kL=193 (ragged kv block). Per-key
    // work = 6*70 = 420; budget 40000 -> chunk 64 -> dispatches over
    // [0,64) [64,128) [128,192) [192,193) — 4, incl. a 1-key ragged tail.
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    fused256_budget_override = 0;
    const single = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    fused256_budget_override = 40_000;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    // Engagement counted, not inferred: the budget must actually split.
    try std.testing.expectEqual(@as(u32, 4), fused256_last_dispatch_count);

    // The carry rides fp32 buffers — the exact register precision — so the
    // chunked result is BIT-IDENTICAL to the single dispatch, not just close.
    const max_diff = try attn256MaxDiff(single, chunked, s);
    try std.testing.expect(max_diff == 0.0);

    // And still correct in absolute terms vs the composed reference.
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const ref_diff = try attn256MaxDiff(chunked, ref, s);
    try std.testing.expect(ref_diff < 0.005);
}

test "fusedSdpa256Prefill: band arm never chunks (single dispatch under a tiny budget)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    fused256_budget_override = 1; // would force max chunking on the causal arm
    defer fused256_budget_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D2);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);

    const fused = (try fusedSdpa256Prefill(s, q, k, v, 1.0, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    // The band arm's block skip already bounds per-dispatch work; chunking it
    // would only add carry traffic. It must stay a single dispatch.
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Unchanged band semantics (same reference as the band parity test).
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));
    const ref = try attn256Reference(q, k, v, 1.0, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpa256Prefill: causal parity at Qwen 24q/4kv geometry (gqa 6, ragged 64-row tile)" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x27B27B);
    const rnd = prng.random();

    // The production Qwen3.6-27B full-attention geometry: 24 q heads over 4
    // kv heads (gqa 6), hd 256. qL=97 exercises a partial 64-row q tile,
    // kL=250 a partial kv block; offset 153 = chunked-prefill alignment.
    const q_shape = [_]c_int{ 1, 24, 97, 256 };
    const kv_shape = [_]c_int{ 1, 4, 250, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Prefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}
