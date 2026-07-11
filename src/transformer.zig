const std = @import("std");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const kv_quant = @import("kv_quant.zig");

pub const KVQuantConfig = kv_quant.KVQuantConfig;
pub const KVQuantScheme = kv_quant.Scheme;

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

// Per-position-state variant of the GDN recurrence kernel, used ONLY on the PLD
// verify forward (small T). Identical math to GDN_KERNEL_SOURCE, but instead of
// writing only the FINAL state it records the state after EVERY timestep into
// `state_seq` ([T, B, Hv, Dv, Dk]). That lets partial-accept rollback restore
// the accepted-position SSM state by slicing — no re-forward of the accepted
// prefix (the costly part on a 48-layer GatedDeltaNet trunk). The decode and
// prefill paths keep using the original single-state kernel, so this adds zero
// cost outside speculative decoding. `seq_stride` = (B*Hv)*Dv*Dk = per-timestep
// element stride into `state_seq` (passed as a scalar, mirroring `T`).
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
    \\  auto t_state = state_seq + t * seq_stride + seq_base;
    \\  for (int i = 0; i < n_per_t; ++i) {
    \\    auto s_idx = n_per_t * dk_idx + i;
    \\    t_state[s_idx] = static_cast<StT>(state[i]);
    \\  }
    \\  q_ += Hk * Dk;
    \\  k_ += Hk * Dk;
    \\  v_ += Hv * Dv;
    \\  y += Hv * Dv;
    \\  g_ += Hv;
    \\  beta_ += Hv;
    \\}
;

var gdn_kernel_seq_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getGdnKernelSeq() !mlx.mlx_fast_metal_kernel {
    if (gdn_kernel_seq_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" };
    const output_names = [_][*:0]const u8{ "y", "state_seq" };
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

const MoeMlpWeights = struct {
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
const QuantParams = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };
const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = [_]?*anyopaque{null} ** BITS_CACHE_CAP,
    vals_bits: [BITS_CACHE_CAP]u8 = [_]u8{0} ** BITS_CACHE_CAP,
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = [_]u8{0} ** BITS_CACHE_CAP,
    vals_mode: [BITS_CACHE_CAP]u8 = [_]u8{0} ** BITS_CACHE_CAP,

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

        if (config.is_encoder_only) return initBert(io, allocator, config, weights, &name_buf, s);
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
        const bias_mandatory = config.quant_bits > 0 and config.quant_mode.hasBiases();
        const emb_s_arr = if (config.quant_bits == 0)
            mlx.mlx_array_new()
        else if (is_nemotron)
            getWeightFmt(weights, &name_buf, "{s}.embeddings.scales", prefix)
        else
            getWeightFmt(weights, &name_buf, "{s}.embed_tokens.scales", prefix);
        const emb_b_arr = if (config.quant_bits == 0)
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
                lm_head_s = if (config.quant_bits == 0) mlx.mlx_array_new() else getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
                lm_head_b = if (config.quant_bits == 0)
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
                        inline for (std.meta.fields(DenseMlpWeights)) |field| {
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
        if (self.compiled_gdn_gate) |cgg| _ = mlx.mlx_closure_free(cgg);
        if (self.gdn_ones_w) |w| _ = mlx.mlx_array_free(w);
        if (self.gdn_q_scale) |q| _ = mlx.mlx_array_free(q);
        if (self.gdn_k_scale) |k| _ = mlx.mlx_array_free(k);
        if (self.ones_hidden) |o| _ = mlx.mlx_array_free(o);
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
    ///     head_dim > 128, which misses MLX's fused kernel (every Gemma-4 and
    ///     Qwen3.5/3.6 checkpoint ships head_dim 256);
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
        const scores: u64 = if (head_dim > 128) @as(u64, n_heads) * chunk_len * total_kv * 2 else 0;
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
                // The sliding window mask limits attention scope.
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, sw);
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
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else if (std.mem.eql(u8, sel_mode, "array")) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", sel_mask, .{ .ctx = null }, self.s));
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

            if (is_prefill and (layer_idx + 1) % std_eval_cadence == 0) {
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
            // source layer's view is what we read.
            const view_layer: u32 = if (is_kv_shared) lw.kv_source.? else li;
            var kv_max: c_int = 0;
            for (ctxs, 0..) |slot_ctx, i| {
                const view = slot_ctx.cache.entries[view_layer].key_view;
                const vshape = mlx.getShape(view);
                const klen: c_int = vshape[2];
                kv_len_buf[i] = klen;
                if (klen > kv_max) kv_max = klen;
            }

            // Pad every slot view to [1, cur_kv_h, kv_max, cur_hd] and concat axis=0.
            const stacked_k = try self.padAndStackBatchedKV(ctxs, view_layer, true, kv_max);
            defer _ = mlx.mlx_array_free(stacked_k);
            const stacked_v = try self.padAndStackBatchedKV(ctxs, view_layer, false, kv_max);
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

    // Pads each slot's KV view (shape [1, kv_h, kv_len_i, head_dim]) to a common
    // kv_max along axis 2 with bf16 zeros and concatenates axis 0 → [N, kv_h, kv_max, head_dim].
    // `key_not_value`: true selects key_view, false selects value_view.
    fn padAndStackBatchedKV(
        self: *const Transformer,
        ctxs: []const *ForwardCtx,
        layer: u32,
        key_not_value: bool,
        kv_max: c_int,
    ) !mlx.mlx_array {
        const pad_axes = [_]c_int{2};
        const pad_value = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(pad_value);

        const padded_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(padded_vec);
        // Track padded arrays so we can free them after the concat.
        var padded_arrs = try self.allocator.alloc(mlx.mlx_array, ctxs.len);
        defer {
            for (padded_arrs) |a| _ = mlx.mlx_array_free(a);
            self.allocator.free(padded_arrs);
        }

        for (ctxs, 0..) |slot_ctx, i| {
            const view = if (key_not_value) slot_ctx.cache.entries[layer].key_view else slot_ctx.cache.entries[layer].value_view;
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

        // PLD spec-decode: thread the per-position SSM capture flag down to the
        // GatedDeltaNet layers (which don't take the ctx). Reset on exit so it
        // never leaks into a non-capturing forward.
        self.spec_capture_ssm = ctx.capture_ssm_seq;
        defer self.spec_capture_ssm = false;

        var h = try self.embedding(token_ids);

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;

        // Qwen3-VL interleaved M-RoPE: build the per-prefill-chunk cos/sin once
        // (shared by every full-attn layer this forward), freed after the loop.
        // Decode (seq_len==1) leaves these null → gatedFullAttnWith uses the
        // scalar offset+delta path.
        if (ctx.mrope_pos != null and is_prefill and @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)) <= ctx.mrope_total) {
            const cs = try self.buildMropeCosSin(ctx, @intCast(offset), @intCast(seq_len));
            ctx.mrope_cos_cur = cs.cos;
            ctx.mrope_sin_cur = cs.sin;
        }
        defer {
            if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
            if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
            ctx.mrope_cos_cur = null;
            ctx.mrope_sin_cur = null;
        }

        // Precompute sliding window masks (Gemma 4)
        var local_prefill_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_prefill_mask);
        var local_decode_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_decode_mask);

        if (is_gemma4 and cfg.has_sliding_window) {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
            if (is_prefill) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, sw);
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
                .full => |fa| if (is_gemma4)
                    try self.gemma4MoeAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, local_prefill_mask, local_decode_mask)
                else
                    try self.gatedFullAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill),
            };
            defer _ = mlx.mlx_array_free(attn_out);

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

            if (is_prefill and (layer_idx + 1) % moe_eval_cadence == 0) {
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

            if (seq_len > 1 and (layer_idx + 1) % hybrid_eval_cadence == 0) {
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
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
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

    // ── Full Attention for MoE models (with optional output gate) ──

    /// Build per-token interleaved-M-RoPE cos/sin [1,1,seq_len,rope_dims] (bf16)
    /// for prompt positions [offset, offset+seq_len) from `ctx.mrope_pos`. The
    /// 3D (t,h,w) divergence only matters at image tokens (prefill); text tokens
    /// have t=h=w so this collapses to ordinary partial RoPE.
    fn buildMropeCosSin(self: *Transformer, ctx: *ForwardCtx, offset: usize, seq_len: usize) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const cfg = &self.config;
        const rope_dims: usize = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const half = rope_dims / 2;
        const pos = ctx.mrope_pos.?;
        const total = ctx.mrope_total;

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
                const pid: f64 = @floatFromInt(pos[axis * total + p]);
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
    fn applyMrope(self: *Transformer, arr: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, rope_dims: c_int) !mlx.mlx_array {
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
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
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
        local_prefill_mask: mlx.mlx_array,
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

        if (!cfg.has_sliding_window) {
            if (is_prefill) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            }
        } else {
            const sw: c_int = @intCast(cfg.sliding_window);
            const total_kv: c_int = offset + seq_len;
            if (is_prefill and is_global) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else if (is_prefill and total_kv <= sw) {
                // Everything fits in the window: the sliding mask degenerates
                // to plain causal. Use the "causal" fast path — same kernel
                // (and reduction order) the mlx-lm/mlx-vlm reference picks for
                // short prompts, which keeps near-tie MoE router decisions
                // from flipping vs the reference (DiffusionGemma parity).
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
            } else if (is_prefill) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask, .{ .ctx = null }, self.s));
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
            // PLD verify pass: emit per-position states so partial-accept
            // rollback needs no re-forward. state_seq: [T, B, Hv, Dv, Dk].
            const state_seq_shape = [_]c_int{ seq_len, batch, num_v_heads, dv, dk };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_seq_shape, 5, .bfloat16));
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
            if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
            try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

            // Stash the full per-position sequence for rollback (takes ownership).
            var state_seq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&state_seq, outputs_vec, 1));
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = state_seq;

            // Continue normal flow with the FINAL state = state_seq[T-1].
            const last: c_int = seq_len - 1;
            const fs_start = [_]c_int{ last, 0, 0, 0, 0 };
            const fs_stop = [_]c_int{ seq_len, batch, num_v_heads, dv, dk };
            const fs_strides = [_]c_int{ 1, 1, 1, 1, 1 };
            var final_sliced = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&final_sliced, state_seq, &fs_start, 5, &fs_stop, 5, &fs_strides, 5, self.s));
            defer _ = mlx.mlx_array_free(final_sliced);
            const fin_shape = [_]c_int{ batch, num_v_heads, dv, dk };
            var final_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&final_state, final_sliced, &fin_shape, 4, self.s));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = final_state;
        } else {
            const state_shape_out = [_]c_int{ batch, num_v_heads, dv, dk };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape_out, 4, .bfloat16));
            // Grid: (32, Dv, B*Hv) threads; threadgroup: (32, 4, 1). Matches mlx-lm.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * num_v_heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

            const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar };
            const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
            defer _ = mlx.mlx_vector_array_free(inputs_vec);

            const gdn_kernel = try getGdnKernel();
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

    fn moeMLP(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        return self.moeMLP2(x, x, mw);
    }

    /// MoE MLP with separate router and expert inputs.
    /// router_x: input for routing (raw hidden states).
    /// expert_x: input for expert computation (possibly normalized).
    fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const cfg = &self.config;
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
        const routed = try self.computeMoeRouting(router_logits);
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
        } else {
            // ── Decode / small-prefill path ──
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

        // Weight by scores: down_out * norm_scores[..., None] → sum over K
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        try mlx.check(mlx.mlx_expand_dims(&scores_exp, norm_scores, -1, self.s)); // [B, S, K, 1]
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_out, scores_exp, self.s));
        var expert_sum = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&expert_sum, weighted, -2, false, self.s)); // [B, S, hidden]

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
            const k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", config.quant_bits);
            const k_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", &config);
            // Gemma 4 MoE: global layers use K=V (no separate v_proj)
            const v_w = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.weight") orelse k_w;
            const v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse k_s;
            const v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse k_b;
            const v_aliases_k = v_w.ctx == k_w.ctx;
            lw.attn = .{ .full = .{
                .q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                .q_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", config.quant_bits),
                .q_b = getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", &config),
                .k_w = k_w,
                .k_s = k_s,
                .k_b = k_b,
                .v_w = v_w,
                .v_s = v_s,
                .v_b = v_b,
                .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", config.quant_bits),
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
            }
        }

        if (config.isMoe() and is_gemma4) {
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
            lw.mlp = .{ .moe = .{
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
        } else if (config.isMoe()) {
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
    inline for (std.meta.fields(FullAttnWeights)) |field| {
        // Dense bf16 full-attn layers carry null-ctx scales/biases — skip those
        // so null arrays don't poison the eval batch. Mirrors the linear/mlp paths.
        const arr = @field(fa, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (std.meta.fields(LinearAttnWeights)) |field| {
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
            inline for (std.meta.fields(MoeMlpWeights)) |field| {
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
            inline for (std.meta.fields(DenseMlpWeights)) |field| {
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
fn computeQuantParams(config: *const ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
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
            inline for (std.meta.fields(BertLayerWeights)) |field| {
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

test "moeRoutingChain produces top-K indices and renormalized softmax weights" {
    const s = mlx.gpuStream();

    // Two rows of router logits over 6 experts. Top-2 of each row is unambiguous:
    //   row 0: experts {0, 3} (logits 10 and 5)
    //   row 1: experts {1, 4} (logits 10 and 5)
    const n_rows: c_int = 2;
    const n_exp: c_int = 6;
    const k: c_int = 2;
    const data = [_]f32{
        10.0, 0.0, 0.0, 5.0, 0.0, 0.0,
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
        var ss = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ss);
        try mlx.check(mlx.mlx_vector_array_get(&ss, outputs_vec, 1));
        // slice [T-1] -> [1,B,Hv,Dv,Dk] -> reshape [B,Hv,Dv,Dk]
        const start = [_]c_int{ T - 1, 0, 0, 0, 0 };
        const stop = [_]c_int{ T, B, Hv, Dv, Dk };
        const strides = [_]c_int{ 1, 1, 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, ss, &start, 5, &stop, 5, &strides, 5, s));
        const fshape = [_]c_int{ B, Hv, Dv, Dk };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, sliced, &fshape, 4, s));
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

test "prefillEvalCadence: small transients keep the coarse cadence" {
    const t = std.testing;
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

test "prefillEvalCadence: big unfused score tensor forces eval-per-layer" {
    const t = std.testing;
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

test "prefillEvalCadence: quantized-KV dequant forces eval-per-layer even at fused head dims" {
    const t = std.testing;
    // hd 128 is fused (zero scores) but denseView rebuilds the FULL cache per
    // layer under --kv-quant: 2 x 600K x 8 x 128 x 2B = 2.46 GB > 2 GiB.
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, true));
    // Same shape with dense fp16 KV: nothing to dequantize, keep coarse.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 600_000, false));
}
