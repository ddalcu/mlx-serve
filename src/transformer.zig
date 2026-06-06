const std = @import("std");
const mlx = @import("mlx.zig");
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
const model_mod = @import("model.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;
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
/// the resulting buffer (free via `SSMCheckpoint.deinit`). Cheap — each layer
/// is a refcount-share, not a copy.
pub fn captureSsmCheckpoint(
    allocator: std.mem.Allocator,
    ssm_entries: []const SSMCacheEntry,
    pos: usize,
) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, 0..) |*src, i| {
        layers[i] = ssmSnapshot(src);
        built = i + 1;
    }
    return .{ .pos = pos, .layers = layers };
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
const QuantParams = struct { bits: u32, group_size: u32 };
const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = [_]?*anyopaque{null} ** BITS_CACHE_CAP,
    vals_bits: [BITS_CACHE_CAP]u8 = [_]u8{0} ** BITS_CACHE_CAP,
    // group_size is always a small power of two in MLX (32, 64, 128). Store as
    // group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = [_]u8{0} ** BITS_CACHE_CAP,

    inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert (bits, group_size) for `key`. Returns false if the probe
    /// window saturated (caller should treat as "fall through to detection"
    /// — but realistically never fires given BITS_CACHE_CAP is 1024 vs ~2k
    /// weights max).
    fn put(self: *QuantParamsCache, key: *anyopaque, bits: u32, group_size: u32) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(bits);
                self.vals_gs_div8[idx] = @intCast(group_size / 8);
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
    vision_embeddings: ?mlx.mlx_array,
    /// Phase 2 (Plan ricky): when true, attention call sites consume the
    /// cache's quantized K/V triples directly via `kv_quant.quantAttention`
    /// instead of dequantizing through `DenseKVView`. Only effective when
    /// the cache scheme is .affine — TurboQuant + .off ignore this flag
    /// (TurboQuant needs the rotation undo step, which the fused path
    /// doesn't yet implement; .off has no quant triple to consume).
    /// Default false → unchanged dense SDPA path.
    kv_attn_fused: bool = false,
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
        const emb_s_arr = (if (is_nemotron)
            getWeightFmtOpt(weights, &name_buf, "{s}.embeddings.scales", prefix)
        else
            getWeightFmtOpt(weights, &name_buf, "{s}.embed_tokens.scales", prefix)) orelse mlx.mlx_array{ .ctx = null };
        const emb_b_arr = (if (is_nemotron)
            getWeightFmtOpt(weights, &name_buf, "{s}.embeddings.biases", prefix)
        else
            getWeightFmtOpt(weights, &name_buf, "{s}.embed_tokens.biases", prefix)) orelse mlx.mlx_array{ .ctx = null };

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
                lm_head_s = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix) orelse mlx.mlx_array{ .ctx = null };
                lm_head_b = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array{ .ctx = null };
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
            ssm_entries = ml.ssm_entries;
            moe_owned_bf16 = ml.owned_bf16;
        } else {
            layers = try initStandardLayers(allocator, config, weights, &name_buf, s);
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
            ple_emb_s = getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.scales", prefix);
            ple_emb_b = getWeightFmt(weights, &name_buf, "{s}.embed_tokens_per_layer.biases", prefix);
            ple_proj_w_g = getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.weight", prefix);
            // per_layer_model_projection may be unquantized (no scales/biases)
            if (getWeightFmtOpt(weights, &name_buf, "{s}.per_layer_model_projection.scales", prefix)) |sc| {
                ple_proj_s_g = sc;
                ple_proj_b_g = getWeightFmt(weights, &name_buf, "{s}.per_layer_model_projection.biases", prefix);
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
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_w);
                    if (lw.q_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_s);
                    if (lw.q_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_b);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_w);
                    if (lw.k_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_s);
                    if (lw.k_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_b);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_w);
                    if (lw.v_s.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_s);
                    if (lw.v_b.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_b);
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
                    if (lw.ple_gate_s) |sc| _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                    if (lw.ple_gate_b) |bi| _ = mlx.mlx_vector_array_append_value(all_vec, bi);
                    if (lw.ple_proj_w) |w| _ = mlx.mlx_vector_array_append_value(all_vec, w);
                    if (lw.ple_proj_s) |sc| _ = mlx.mlx_vector_array_append_value(all_vec, sc);
                    if (lw.ple_proj_b) |bi| _ = mlx.mlx_vector_array_append_value(all_vec, bi);
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
        // Resolve (bits, group_size) per weight. Most weights inherit the global
        // config; per-weight overrides land in the cache at bind time so qmatmul
        // calls see the correct params without changing the call sites.
        const qp = self.quantParamsFor(w, sc);
        return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, self.s);
    }

    /// Resolve per-weight quant bits — convenience for callers that only need
    /// bits (e.g. embedding gather) and use `config.quant_group_size` directly.
    inline fn bitsFor(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
        _ = group_size; // accepted for API compatibility; we always use the cached value
        return self.quantParamsFor(w, sc).bits;
    }

    /// Resolve per-weight (bits, group_size) with a lazy cache keyed by the
    /// scales array pointer. First touch computes bits from shapes against the
    /// global config's group_size (~4 FFI calls); subsequent calls are a single
    /// pointer compare. Pre-populated entries take precedence over the global
    /// default. Thread-safety: generation is single-threaded.
    inline fn quantParamsFor(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array) QuantParams {
        const default_gs = self.config.quant_group_size;
        const key_raw = sc.ctx orelse return .{ .bits = self.config.quant_bits, .group_size = default_gs };
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
                };
            }
            if (entry == null) {
                const dp = detectQuantParams(w, sc, default_gs);
                cache.keys[idx] = key_raw;
                cache.vals_bits[idx] = @intCast(dp.bits);
                cache.vals_gs_div8[idx] = @intCast(dp.group_size / 8);
                return .{ .bits = dp.bits, .group_size = dp.group_size };
            }
        }
        // Probe window saturated — fall through to direct detect, no cache write.
        return detectQuantParams(w, sc, default_gs);
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
            // Plain bf16 embedding: no scales/biases, no dequant — the gathered
            // rows are already the embedding vectors. Cast to bf16 for parity
            // with the dequantized path's output dtype.
            try mlx.check(mlx.mlx_astype(&emb, taken_w, .bfloat16, self.s));
        } else {
            var taken_s = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_s);
            try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
            const emb_qp = self.quantParamsFor(self.emb_w, self.emb_s);
            const emb_desc = quantDescFor(self.emb_w, self.emb_s, self.emb_b, emb_qp.bits, emb_qp.group_size);

            var taken_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(taken_b);
            // mxfp8 has no bias term: pass a null handle and skip the bias gather.
            const dq_biases = if (isMxQuant(self.emb_b)) mlx.mlx_array{ .ctx = null } else blk: {
                try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));
                break :blk taken_b;
            };

            try mlx.check(mlx.mlx_dequantize(
                &emb,
                taken_w,
                taken_s,
                dq_biases,
                mlx.mlx_optional_int.some(@intCast(emb_desc.group_size)),
                mlx.mlx_optional_int.some(@intCast(emb_desc.bits)),
                emb_desc.mode,
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

        // Position IDs: [0, 1, 2, ..., seq_len-1]
        var pos_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_ids);
        try mlx.check(mlx.mlx_arange(&pos_ids, 0, @as(f64, @floatFromInt(seq_len)), 1, .int32, self.s));

        const pos_emb = try self.dequantTake(self.bert_pos_w, self.bert_pos_s, self.bert_pos_b, pos_ids);
        defer _ = mlx.mlx_array_free(pos_emb);

        // Token type IDs: all zeros
        var toktype_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(toktype_ids);
        try mlx.check(mlx.mlx_zeros(&toktype_ids, &flat_shape, 1, .int32, self.s));

        const toktype_emb = try self.dequantTake(self.bert_toktype_w, self.bert_toktype_s, self.bert_toktype_b, toktype_ids);
        defer _ = mlx.mlx_array_free(toktype_emb);

        // Sum: word + position + token_type
        var wp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wp);
        try mlx.check(mlx.mlx_add(&wp, word_emb, pos_emb, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, wp, toktype_emb, self.s));

        // Reshape to [B, S, H]
        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var reshaped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(reshaped);
        try mlx.check(mlx.mlx_reshape(&reshaped, sum, &out_shape, 3, self.s));

        // LayerNorm
        return self.layerNorm(reshaped, self.bert_emb_norm_w, self.bert_emb_norm_b);
    }

    fn dequantTake(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !mlx.mlx_array {
        var tw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tw);
        try mlx.check(mlx.mlx_take_axis(&tw, w, ids, 0, self.s));
        var ts = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ts);
        try mlx.check(mlx.mlx_take_axis(&ts, sc, ids, 0, self.s));
        var tb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tb);
        try mlx.check(mlx.mlx_take_axis(&tb, bi, ids, 0, self.s));
        var result = mlx.mlx_array_new();
        const qp = self.quantParamsFor(w, sc);
        try mlx.check(mlx.mlx_dequantize(
            &result,
            tw,
            ts,
            tb,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            "affine",
            .{}, // global_scale (null)
            .{ .value = .bfloat16, .has_value = true },
            self.s,
        ));
        return result;
    }

    fn forwardBertWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        _ = ctx; // BERT is encoder-only, no per-request KV state
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

            // Bidirectional attention (no causal mask)
            var attn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
                &attn,
                q_t,
                k_t,
                v_t,
                scale,
                "",
                mlx.mlx_array_new(),
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
            const q = try self.qmatmul(normed, lw.q_w, lw.q_s, lw.q_b);
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
                const own_k = try self.qmatmul(normed, lw.k_w, lw.k_s, lw.k_b);
                defer _ = mlx.mlx_array_free(own_k);
                const own_v = if (lw.k_eq_v)
                    own_k
                else
                    try self.qmatmul(normed, lw.v_w, lw.v_s, lw.v_b);
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

            if (is_prefill and (layer_idx + 1) % EVAL_EVERY_N_LAYERS == 0) {
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

        if (self.embedding_mode) return final_normed;
        var logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
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
            const q = try self.qmatmul(normed, lw.q_w, lw.q_s, lw.q_b);
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
                const own_k = try self.qmatmul(normed, lw.k_w, lw.k_s, lw.k_b);
                defer _ = mlx.mlx_array_free(own_k);
                const own_v = if (lw.k_eq_v)
                    own_k
                else
                    try self.qmatmul(normed, lw.v_w, lw.v_s, lw.v_b);
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

        var logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
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
        const is_gemma4 = std.mem.eql(u8, cfg.model_type, "gemma4");

        var h = try self.embedding(token_ids);

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;

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
                // Gemma 4 layer structure (from HF reference):
                //   h = residual + post_attn_norm(attn(input_norm(residual)))
                //   residual = h
                //   shared = post_ff_norm_1(mlp(pre_ff_norm(h)))
                //   experts = post_ff_norm_2(moe(pre_ff_norm_2(residual)))
                //   h = residual + post_ff_norm(shared + experts)
                //   h *= layer_scalar

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

                // Layer scalar
                if (lw.layer_scalar) |ls| {
                    var h_scaled = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_multiply(&h_scaled, h, ls, self.s));
                    _ = mlx.mlx_array_free(h);
                    h = h_scaled;
                }
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

            if (is_prefill and (layer_idx + 1) % MOE_EVAL_EVERY_N_LAYERS == 0) {
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

        if (self.embedding_mode) return final_normed;
        var logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
        _ = mlx.mlx_array_free(final_normed);

        // Gemma 4: logit softcapping — tanh(logits / cap) * cap
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

            if (seq_len > 1 and (layer_idx + 1) % MOE_EVAL_EVERY_N_LAYERS == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }

        }

        ctx.moe_seq_offset.* += @intCast(seq_len);

        // Final norm (absent for LFM2)
        if (cfg.has_final_norm) {
            const final_normed = try self.rmsNorm(h, self.final_norm);
            _ = mlx.mlx_array_free(h);
            if (self.embedding_mode) return final_normed;
            const logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
            _ = mlx.mlx_array_free(final_normed);
            return logits;
        } else {
            if (self.embedding_mode) return h;
            const logits = try self.qmatmul(h, self.lm_head_w, self.lm_head_s, self.lm_head_b);
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
        self.embedding_mode = true;
        defer self.embedding_mode = false;
        return self.forward(token_ids);
    }

    // ── Full Attention for MoE models (with optional output gate) ──

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

        // Partial RoPE
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));

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
            } else if (is_prefill) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask, .{ .ctx = null }, self.s));
            } else if (is_global) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            } else if (@as(c_int, @intCast(ctx.cache.seqLen(layer))) <= sw) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
            } else {
                _ = total_kv;
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
        const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
        const inv_sqrt_scale = @sqrt(inv_scale);
        const inv_scale_sq = bf16Scalar(inv_scale, self.s);
        defer _ = mlx.mlx_array_free(inv_scale_sq);
        const inv_sqrt_sc = bf16Scalar(inv_sqrt_scale, self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt_sc);

        // Parameter-free RMS norm: use ones weight (mlx-c requires non-empty array)
        const ones_shape = [_]c_int{dk};
        var ones_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones_w);
        try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, self.s));

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

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
        // Cast A_log to float32 for stability
        var A_log_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_log_f32);
        try mlx.check(mlx.mlx_astype(&A_log_f32, la.A_log, .float32, self.s));
        var exp_A = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_A);
        try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, self.s));

        // softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
        var a_plus_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_plus_dt);
        try mlx.check(mlx.mlx_add(&a_plus_dt, a_proj, la.dt_bias, self.s));
        var a_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_f32);
        try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, self.s));
        var exp_a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_a);
        try mlx.check(mlx.mlx_exp(&exp_a, a_f32, self.s));
        var sp_inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sp_inner);
        try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, self.s));

        // -exp(A_log) * softplus(...)
        var neg_decay = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg_decay);
        try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, self.s));
        var neg_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg_neg);
        try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, self.s));
        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, self.s)); // [B, S, Hv]
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_astype(&g, g_f32, .bfloat16, self.s));

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

        const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        const y_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        const state_shape_out = [_]c_int{ batch, num_v_heads, dv, dk };

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
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

        const gdn_kernel = try getGdnKernel();
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));

        if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;

        var y_bthd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_bthd);
        try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

        var new_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&new_state, outputs_vec, 1));
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = new_state;

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
    /// One expert projection in the MoE dispatch. Quantized experts go through
    /// `gather_qmm` (transpose_w=true, consuming stored [experts, out, in]);
    /// plain-bf16 experts (null scales) go through dense `gather_mm`, which does
    /// not transpose and therefore needs [experts, in, out] — the layout that
    /// `maybeTransposeForBf16` produces at load. Gather/sort semantics are
    /// identical across both ops.
    fn expertGatherMatmul(
        self: *const Transformer,
        res: *mlx.mlx_array,
        x: mlx.mlx_array,
        w: mlx.mlx_array,
        sc: mlx.mlx_array,
        q: QuantDesc,
        lhs_indices: mlx.mlx_array,
        rhs_indices: mlx.mlx_array,
        sorted_indices: bool,
    ) !void {
        if (sc.ctx == null) {
            try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_indices, rhs_indices, sorted_indices, self.s));
        } else {
            try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, q.biases, lhs_indices, rhs_indices, true, mlx.mlx_optional_int.some(@intCast(q.group_size)), mlx.mlx_optional_int.some(@intCast(q.bits)), q.mode, sorted_indices, self.s));
        }
    }

    fn moeMLP2(self: *Transformer, router_x: mlx.mlx_array, expert_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const cfg = &self.config;
        const gs = cfg.quant_group_size;
        // Per-expert-weight bits: Gemma-4 MoE has 4-bit experts but 8-bit router/shared,
        // so we can't use a single layer-wide bits value — resolve per weight.
        const gate_bits = self.bitsFor(mw.switch_gate_w, mw.switch_gate_s, gs);
        const up_bits = self.bitsFor(mw.switch_up_w, mw.switch_up_s, gs);
        const down_bits = self.bitsFor(mw.switch_down_w, mw.switch_down_s, gs);

        // Per-expert quant descriptors. mxfp8 experts have scales but no biases,
        // and need mode="mxfp8"/bits=8 rather than the affine path.
        const gate_q = quantDescFor(mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, gate_bits, gs);
        const up_q = quantDescFor(mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, up_bits, gs);
        const down_q = quantDescFor(mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, down_bits, gs);

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

            const rqp = self.quantParamsFor(mw.router_w, mw.router_s);
            router_logits = try qmatmulBits(normed_input, mw.router_w, mw.router_s, mw.router_b, rqp.bits, rqp.group_size, self.s);
        } else {
            // Qwen3.5: direct projection
            const rqp = self.quantParamsFor(mw.router_w, mw.router_s);
            router_logits = try qmatmulBits(router_x, mw.router_w, mw.router_s, mw.router_b, rqp.bits, rqp.group_size, self.s);
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
        // bf16 experts (null scales) use dense `gather_mm`, whose decode-path
        // index broadcasting against the 5-D [B,S,1,1,D] layout is unverified;
        // the sorted path's flat [N,1,D] layout is unambiguous, so always take
        // it for bf16. Quantized experts keep the original heuristic.
        const experts_bf16 = mw.switch_gate_s.ctx == null;
        const do_sort = experts_bf16 or S > 1 or total_inds >= 64;
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
            try self.expertGatherMatmul(&gate_out_3d, x_rep, mw.switch_gate_w, mw.switch_gate_s, gate_q, no_idx, sorted_inds, true);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_3d, self.s));

            var up_out_3d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out_3d);
            try self.expertGatherMatmul(&up_out_3d, x_rep, mw.switch_up_w, mw.switch_up_s, up_q, no_idx, sorted_inds, true);
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
            try self.expertGatherMatmul(&down_3d, act_exp, mw.switch_down_w, mw.switch_down_s, down_q, no_idx, sorted_inds, true);
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
            try self.expertGatherMatmul(&gate_out_5d, x_exp, mw.switch_gate_w, mw.switch_gate_s, gate_q, no_idx, inds, false);
            var gate_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_out);
            try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_5d, self.s));

            var up_out_5d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(up_out_5d);
            try self.expertGatherMatmul(&up_out_5d, x_exp, mw.switch_up_w, mw.switch_up_s, up_q, no_idx, inds, false);
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
            try self.expertGatherMatmul(&down_5d, act_exp, mw.switch_down_w, mw.switch_down_s, down_q, no_idx, inds, false);
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
        const seg_qp = self.quantParamsFor(seg_w, mw.shared_expert_gate_s.?);
        const sh_gate_logit = try qmatmulBits(expert_x, seg_w, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, seg_qp.bits, seg_qp.group_size, self.s);
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

fn initStandardLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) ![]LayerWeights {
    log.info("Precomputing layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

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
            const k_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight");
            lw.k_norm = if (config.norm_has_offset) try addOne(k_norm_raw, s) else k_norm_raw;
        } else {
            lw.q_norm = null;
            lw.k_norm = null;
        }

        lw.q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight");
        lw.q_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.q_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.biases") orelse mlx.mlx_array{ .ctx = null };
        lw.k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
        lw.k_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.k_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.biases") orelse mlx.mlx_array{ .ctx = null };
        // Gemma 4 (31B): full_attention layers share V with K (no v_proj weight stored).
        // Sliding_attention layers still have separate V.
        lw.k_eq_v = config.attention_k_eq_v and config.isGlobalLayer(li);
        if (lw.k_eq_v) {
            lw.v_w = lw.k_w;
            lw.v_s = lw.k_s;
            lw.v_b = lw.k_b;
        } else {
            lw.v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
            lw.v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse mlx.mlx_array{ .ctx = null };
            lw.v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse mlx.mlx_array{ .ctx = null };
        }
        lw.o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight");
        lw.o_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.biases") orelse mlx.mlx_array{ .ctx = null };

        lw.gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight");
        lw.gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate_proj.biases") orelse mlx.mlx_array{ .ctx = null };
        lw.up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight");
        lw.up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.up_proj.biases") orelse mlx.mlx_array{ .ctx = null };
        lw.down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight");
        lw.down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.scales") orelse mlx.mlx_array{ .ctx = null };
        lw.down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.down_proj.biases") orelse mlx.mlx_array{ .ctx = null };

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
            lw.ple_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_input_gate.scales");
            lw.ple_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_input_gate.biases");
            lw.ple_proj_w = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_projection.weight");
            lw.ple_proj_s = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_projection.scales");
            lw.ple_proj_b = getLayerWeightOpt(weights, name_buf, prefix, li, "per_layer_projection.biases");
            lw.ple_norm = getLayerWeightOpt(weights, name_buf, prefix, li, "post_per_layer_input_norm.weight");
        }

        // Gemma 4: KV sharing source layer
        lw.kv_source = config.getKVSourceLayer(li);
    }
    return layers;
}

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var w_t = mlx.mlx_array_new();
    // Swap the last two axes. 2-D Linear weights become [in, out]; 3-D stacked
    // MoE expert weights [experts, out, in] keep the expert axis and become
    // [experts, in, out]. Higher ranks fall back to the 2-D case (unused today).
    const ndim = mlx.getShape(w).len;
    if (ndim == 3) {
        const perm = [_]c_int{ 0, 2, 1 };
        try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, 3, s));
    } else {
        const perm = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, 2, s));
    }
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
    if (sc.ctx != null) return;
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
    const is_gemma4 = std.mem.eql(u8, config.model_type, "gemma4");

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
        lw.shared_mlp = null;

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
            const k_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.scales") orelse mlx.mlx_array{ .ctx = null };
            const k_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.k_proj.biases") orelse mlx.mlx_array{ .ctx = null };
            // Gemma 4 MoE: global layers use K=V (no separate v_proj)
            const v_w = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.weight") orelse k_w;
            const v_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.scales") orelse k_s;
            const v_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.v_proj.biases") orelse k_b;
            lw.attn = .{ .full = .{
                .q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                .q_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.scales") orelse mlx.mlx_array{ .ctx = null },
                .q_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.q_proj.biases") orelse mlx.mlx_array{ .ctx = null },
                .k_w = k_w,
                .k_s = k_s,
                .k_b = k_b,
                .v_w = v_w,
                .v_s = v_s,
                .v_b = v_b,
                .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.scales") orelse mlx.mlx_array{ .ctx = null },
                .o_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.o_proj.biases") orelse mlx.mlx_array{ .ctx = null },
                .q_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight"),
                .k_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight"),
            } };
        }

        if (config.isMoe() and is_gemma4) {
            // Gemma 4 MoE: different weight naming, Sigma-MoE routing, no shared expert gate.
            // Each `*_s`/`*_b` is loaded optionally for Unsloth Dynamic compatibility —
            // bf16 layers carry only the weight, no scales/biases. The post-construction
            // `maybeTransposeForBf16` calls are no-ops for already-quantized weights.
            lw.mlp = .{ .moe = .{
                .router_w = getLayerWeight(weights, name_buf, prefix, li, "router.proj.weight"),
                .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.scales") orelse mlx.mlx_array_new(),
                .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "router.proj.biases") orelse mlx.mlx_array_new(),
                .router_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.scale"),
                .per_expert_scale = getLayerWeightOpt(weights, name_buf, prefix, li, "router.per_expert_scale"),
                .switch_gate_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.weight"),
                .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.scales") orelse mlx.mlx_array_new(),
                .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.gate_proj.biases") orelse mlx.mlx_array_new(),
                .switch_up_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.weight"),
                .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.scales") orelse mlx.mlx_array_new(),
                .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.up_proj.biases") orelse mlx.mlx_array_new(),
                .switch_down_w = getLayerWeight(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.weight"),
                .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.scales") orelse mlx.mlx_array_new(),
                .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "experts.switch_glu.down_proj.biases") orelse mlx.mlx_array_new(),
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
            // Qwen3.5 MoE. Each `*_s`/`*_b` is loaded optionally — Unsloth Dynamic
            // checkpoints (e.g. Qwen3.6-A3B UD) leave the router (`mlp.gate`) and the
            // shared-expert gate (`mlp.shared_expert_gate`) as plain bf16, with no
            // scales/biases. The `maybeTransposeForBf16` calls below pre-transpose
            // bf16 weights from `[out, in]` → `[in, out]` so `qmatmulBits` can
            // dispatch to plain `mlx_matmul`. They no-op on already-quantized weights.
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
                .shared_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight"),
                .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
                .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
                .shared_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight"),
                .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
                .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
                .shared_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight"),
                .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
                .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
                .shared_expert_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
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
                    .in_proj_b = getLayerWeight(weights, name_buf, prefix, li, "conv.in_proj.biases"),
                    .conv_w = getLayerWeight(weights, name_buf, prefix, li, "conv.conv.weight"),
                    .out_proj_w = getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.weight"),
                    .out_proj_s = getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.scales"),
                    .out_proj_b = getLayerWeight(weights, name_buf, prefix, li, "conv.out_proj.biases"),
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
                        .q_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.biases"),
                        .k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight"),
                        .k_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.scales"),
                        .k_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.biases"),
                        .v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight"),
                        .v_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.scales"),
                        .v_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.biases"),
                        .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.weight"),
                        .o_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.scales"),
                        .o_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.out_proj.biases"),
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
                .gate_b = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w1.biases"),
                .up_w = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w3.weight"),
                .up_s = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w3.scales"),
                .up_b = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w3.biases"),
                .down_w = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w2.weight"),
                .down_s = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w2.scales"),
                .down_b = getLayerWeight(weights, name_buf, prefix, li, "feed_forward.w2.biases"),
            };
        } else {
            lw.mlp = null;
        }
    }

    return .{ .hybrid_layers = hybrid_layers, .ssm_entries = ssm_entries };
}

fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (std.meta.fields(FullAttnWeights)) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        // mxfp8 / plain-bf16 layers carry null-ctx scales/biases — skip those.
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

/// Jointly recover (bits, group_size) for an affine-quantized weight from its
/// packed-weight and scales shapes, without trusting a globally-configured
/// group_size. Mixed-precision checkpoints (e.g. mxfp8 experts with affine
/// router) carry per-module group_size overrides that the loader does not
/// thread through, so a single global value mis-detects bits for the modules
/// that differ.
///
/// For affine packing the packed weight has `w_cols = in * bits / 32` and the
/// scales have `s_cols = in / gs`, hence `w_cols * 32 / s_cols = bits * gs`.
/// That product is fixed by the shapes; we pick the standard (bits, gs) pair
/// that satisfies it, preferring higher bit widths. Falls back to the passed
/// `group_size` with the legacy formula if no standard pair matches.
fn detectQuantParams(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return .{ .bits = 4, .group_size = group_size };
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return .{ .bits = 4, .group_size = group_size };
    const product = (w_cols * 32) / s_cols; // == bits * gs
    // Prefer wider bits; gs must be a standard group size.
    const bit_candidates = [_]u32{ 8, 4, 2 };
    for (bit_candidates) |b| {
        if (product % b != 0) continue;
        const gs = product / b;
        if (gs == 32 or gs == 64 or gs == 128) return .{ .bits = b, .group_size = gs };
    }
    // No standard pair — keep the configured gs and derive bits from it.
    return .{ .bits = detectQuantBits(w, sc, group_size), .group_size = group_size };
}

/// MoE routing chain (negate→argpartition→slice→softmax→take→sum→expand→divide).
/// Free-function variant of `Transformer.moeRoutingUncompiled` so unit tests can
/// exercise the pure subgraph without constructing a full Transformer. Returns
/// owned `inds` (int32, [..., k]) and `norm_scores` (bf16, [..., k]) — caller
/// must free both.
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

fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    // Plain BF16 weight: scales array is unset. Used by mixed-precision Unsloth
    // Dynamic checkpoints that leave a subset of layers (e.g. linear_attn
    // projections in Qwen3.6 UD) unquantized. The weight is pre-transposed at
    // load to [in, out] so a single mlx_matmul does the contraction.
    if (sc.ctx == null) {
        var fp_result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&fp_result, x, w, s));
        return fp_result;
    }

    // Detect mxfp8 mode: weight has scales but no affine bias term (either a
    // null handle or an empty placeholder array from an absent optional load).
    const is_mxfp = isMxQuant(bi);

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

/// True when a quantized weight carries no affine bias term — i.e. it is stored
/// in a microscaling format (mxfp8), which has `scales` but no `biases`. Both an
/// explicit null handle and an empty placeholder array (produced by
/// `mlx_array_new()` on an absent optional weight) count as "no biases".
fn isMxQuant(bi: mlx.mlx_array) bool {
    return bi.ctx == null or mlx.getShape(bi).len == 0;
}

const QuantDesc = struct {
    mode: [*:0]const u8,
    bits: u32,
    group_size: u32,
    biases: mlx.mlx_array,
};

/// Resolve (mode, bits, group_size, biases) for a quantized weight, mirroring
/// the inference in `qmatmulBits`. Affine weights keep the caller-supplied
/// `aff_bits`/`aff_gs` and their real biases. Microscaling (mxfp8) weights are
/// 8-bit with group_size inferred from the scales/weight column ratio and a
/// null bias handle (mxfp8 has no bias term).
fn quantDescFor(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, aff_bits: u32, aff_gs: u32) QuantDesc {
    // Affine (real biases) or unquantized (no scales at all): keep the affine
    // params and the caller-supplied biases array. Only a weight with scales but
    // no biases is microscaling.
    if (sc.ctx == null or !isMxQuant(bi))
        return .{ .mode = "affine", .bits = aff_bits, .group_size = aff_gs, .biases = bi };

    const mxfp_bits: u32 = 8;
    const s_shape = mlx.getShape(sc);
    const w_shape = mlx.getShape(w);
    const mxfp_gs: u32 = if (s_shape.len >= 2 and w_shape.len >= 2) blk: {
        const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
        const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
        if (s_cols > 0) break :blk (w_cols * 32) / (s_cols * mxfp_bits);
        break :blk 32;
    } else 32;
    return .{ .mode = "mxfp8", .bits = mxfp_bits, .group_size = mxfp_gs, .biases = .{ .ctx = null } };
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

    try testing.expect(cache.put(a.ctx.?, 4, 32));
    try testing.expect(cache.put(b.ctx.?, 4, 64));
    try testing.expect(cache.put(c.ctx.?, 8, 128));

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
    const got = try qmatmulBits(x, w_t, null_sc, null_bi, 4, 64, s);
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
