//! sdk/blocks/attention/kv_view.
//! Owns DenseKVView, SlidingView.

const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("kv_quant.zig");

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
    /// (scheme == .off). Read-only; the cache owns these handles.
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
    /// Axis-2 capacity of the triples' parent buffers (all six row-contiguous, grown in lockstep); 0 = unknown.
    /// The slot kernel strides by it: a lazy view reports contiguous strides until evaluated.
    kv_cap: c_int = 0,

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

/// What a sliding layer's K/V view looks like for THIS forward: the `max_kv`
/// to hand `KVCache.update`, the length the view will actually have, and
/// whether the fused hd-256 kernel is the thing doing the band masking.
///
/// ONE chokepoint for the policy — every arch path asks it rather than
/// re-deriving the predicate, which is how the standard and MoE paths drifted
/// apart in the first place.
pub const SlidingView = struct {
    /// `max_kv` for `KVCache.update`; 0 = hand back the whole cache.
    span: u32,
    /// The K/V length a sliding layer will see. EVERY mask builder must be
    /// handed this, not `total_kv`: they derive the query offset relatively
    /// (`offset_val = kv_len - q_len`), so a trimmed view with the trimmed
    /// length is correct and a trimmed view with the full length is not.
    kv_len: c_int,
    /// The fused hd-256 kernel masks the band in-kernel off ABSOLUTE positions
    /// (`q_off = kL - qL`, `(row_pos - col) >= SW`), so a trimmed view would
    /// shift its band. When true the trim is declined AND the explicit mask
    /// build is skipped — the same predicate decides both.
    band_in_kernel: bool,
};
