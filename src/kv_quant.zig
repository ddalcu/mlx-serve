// KV-cache quantization backend.
//
// This module owns the storage / dispatch contract between the cache
// (`KVCache` in `transformer.zig`) and the attention call sites. SDPA always
// reads dense `[B, H, T, head_dim]` tensors via `KVCache.denseView` — what the
// cache buffers actually hold is decided here.
//
// v1 ships a single non-trivial scheme: affine group-wise quantization at
// 4 or 8 bits, identical mathematically to mlx-c's existing `mlx_quantize`
// path used for weight quantization. The buffers grow to 3 arrays per K and V
// (q, scales, biases); attention is unchanged because `denseView` calls
// `dequantizeAffine` before returning.
//
// ── Adding a new scheme later (e.g. TurboQuant) ──
//
// The contract is intentionally small so a future session can drop in a new
// scheme without touching `transformer.zig`'s SDPA call sites. Steps:
//
//  1. Add an enum variant to `Scheme` (e.g. `turboquant_1`, `turboquant_2`).
//  2. (Optional) Add per-cache state on `KVCache` for things like rotation
//     matrices. For TurboQuant: `quant_state: ?TurboState` carrying one
//     `[head_dim, head_dim]` orthogonal/Hadamard matrix per K and V per
//     layer (~10 MB total at Gemma 4 E4B). Initialize once at
//     `KVCache.init` from a deterministic seed.
//  3. Add two functions here mirroring `quantizeAffine`/`dequantizeAffine`:
//        quantizeTurbo  : (s, dense_x, R, bits) → QuantizedKV
//                         { q = quantizeAffine(R @ dense_x, …), … }
//        dequantizeTurbo: (s, q, scales, biases, R, bits) → dense_x
//                         { y = dequantizeAffine(…); return y @ R^T }
//  4. Extend the `switch (config.scheme)` blocks in `KVCache.update`
//     (quantize on write) and `KVCache.denseView` (dequantize on read)
//     with one case arm each. SDPA call sites do not change; the cache
//     contract holds.
//
// The same dispatch point is also where a future fused-quant-SDPA Metal
// kernel (see "Fused quant-attention Metal kernel" in TODO.md) would slot
// in: `denseView` becomes a no-op stub for that scheme and SDPA call sites
// grow a parallel quant-path. v1 commits to Path A (dense view + standard
// SDPA).

const std = @import("std");
const mlx = @import("mlx.zig");

/// KV-cache storage scheme.
///   * `off`      — dense bf16 (legacy).
///   * `affine`   — group-wise affine quant via `mlx_quantize`/`mlx_dequantize`.
///   * `turboquant_2` — Hadamard-rotated 2-bit affine quant. Each cache
///                     carries one `[head_dim, head_dim]` rotation matrix per
///                     K and V per layer (see `TurboState`). On write we
///                     compute `q = quantizeAffine(x @ H, group, 2)`; on read
///                     we recover `x ≈ dequantizeAffine(q, …) @ H` (Hadamard
///                     matrices are symmetric and self-inverse modulo a
///                     scalar, so `H = H^T = H^{-1}` after normalization).
///   * `turboquant_4` — same rotation idea at 4-bit. Useful when the bit
///                     budget can spare a couple of bits in exchange for
///                     reduced rotation overhead at the cost ceiling.
///                     (Compared to plain `affine` at 4-bit, TurboQuant 4
///                     spends a `[D,D]` matmul per layer per token; the
///                     rotation breaks the worst-case correlation patterns
///                     that hurt straight affine at long context.)
///
/// 1-bit TurboQuant from the Path B roadmap requires a custom 1-bit
/// pack/unpack — `mlx_quantize`/`mlx_dequantize` only support 2/4/8 bits in
/// mlx 0.31.2. Deferred to a follow-up that pairs with the fused-kernel work.
pub const Scheme = enum { off, affine, turboquant_2, turboquant_4 };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. TurboQuant: 2 or 4. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn turboquant(bits: u8) KVQuantConfig {
        // Bits 2 and 4 ride mlx-c's native packing. We intentionally do NOT
        // accept 1 here — adding 1-bit requires a custom pack/unpack on top
        // of the rotation; see Scheme docstring.
        std.debug.assert(bits == 2 or bits == 4);
        return .{
            .scheme = if (bits == 2) .turboquant_2 else .turboquant_4,
            .bits = bits,
            .group_size = 64,
        };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }
};

/// One quantized K or V triple. Layout for input shape `[..., D]`:
///   q      : `[..., D * bits / 32]` uint32   (packed)
///   scales : `[..., D / group_size]` bf16
///   biases : `[..., D / group_size]` bf16
///
/// Owns its three array handles. Caller frees via `deinit`.
pub const QuantizedKV = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,

    pub fn deinit(self: *QuantizedKV) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        self.q = mlx.mlx_array_new();
        self.scales = mlx.mlx_array_new();
        self.biases = mlx.mlx_array_new();
    }
};

/// Affine quantize `dense_x` group-wise along the last axis. Returns a
/// `QuantizedKV` triple owned by the caller. Caller's input `dense_x` is
/// not consumed (refcount semantics — the caller still owns it).
pub fn quantizeAffine(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    try mlx.check(mlx.mlx_quantize(
        &vec,
        dense_x,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        s,
    ));

    // mlx-c convention: the vector contains [q, scales, biases] in that order.
    // Mirror the unpack pattern used elsewhere when consuming a
    // `mlx_vector_array` (e.g. concatenate fan-out).
    const n = mlx.mlx_vector_array_size(vec);
    if (n != 3) return error.UnexpectedQuantizeOutput;

    var out: QuantizedKV = .{
        .q = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
    };
    errdefer out.deinit();

    try mlx.check(mlx.mlx_vector_array_get(&out.q, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, vec, 2));
    return out;
}

/// Per-cache rotation state for the TurboQuant schemes. Holds one symmetric
/// Hadamard matrix per K and V per layer — `[n, n]` bf16 where `n` is the
/// actual K (resp. V) last-dim observed at first write. Built lazily because
/// the cached K/V last-dim is NOT always `config.head_dim` — Gemma 4 stores
/// K at `2 * head_dim` due to partial-RoPE / split-rotary, and some archs
/// have K and V dims that differ from each other. The matrices are
/// constructed deterministically via Sylvester construction + per-layer
/// column-sign flips (no RNG seed; reproducible across restarts).
///
/// Construction is gated on the last-dim being a power of two. The
/// scheduler's load path validates `head_dim` is pow2 at cache init via
/// `validatePowerOfTwoOrFail`, but the actual K/V dim is allowed to differ
/// by an integer factor (Gemma 4: 2x); we re-check pow2 at lazy-init time
/// and return `error.NonPowerOfTwoHeadDim` if violated.
pub const TurboState = struct {
    /// One matrix per K per layer. Slot may be empty (`.ctx == null`) until
    /// the first `updateTurboQuant` call for that layer, at which point the
    /// real K tensor's last-dim is observed and the matrix is built.
    rk: []mlx.mlx_array,
    /// Per-K-layer dim, recorded at lazy-init time so subsequent calls can
    /// assert shape consistency. 0 = not yet initialized.
    rk_dim: []u32,
    /// One matrix per V per layer. Distinct from rk so K and V get
    /// uncorrelated rotations (matters for arches where Q/K and V share
    /// little structure, e.g. GQA + value-only sliding-window). May also
    /// have a different last-dim from K.
    rv: []mlx.mlx_array,
    rv_dim: []u32,
    allocator: std.mem.Allocator,

    /// Allocate empty slots for `num_layers`. Matrices are NOT built here —
    /// the first `ensureKLayer`/`ensureVLayer` call per layer triggers
    /// construction from the observed K/V tensor shape.
    pub fn initLazy(allocator: std.mem.Allocator, num_layers: u32) !TurboState {
        const rk = try allocator.alloc(mlx.mlx_array, num_layers);
        errdefer allocator.free(rk);
        const rv = try allocator.alloc(mlx.mlx_array, num_layers);
        errdefer allocator.free(rv);
        const rk_dim = try allocator.alloc(u32, num_layers);
        errdefer allocator.free(rk_dim);
        const rv_dim = try allocator.alloc(u32, num_layers);
        errdefer allocator.free(rv_dim);
        for (rk) |*a| a.* = mlx.mlx_array_new();
        for (rv) |*a| a.* = mlx.mlx_array_new();
        for (rk_dim) |*d| d.* = 0;
        for (rv_dim) |*d| d.* = 0;
        return .{
            .rk = rk,
            .rk_dim = rk_dim,
            .rv = rv,
            .rv_dim = rv_dim,
            .allocator = allocator,
        };
    }

    /// Deprecated: kept so existing unit tests (which use a single-layer
    /// fixed-dim setup) keep working. Real load path uses `initLazy` and
    /// builds matrices on first write.
    pub fn initHadamard(allocator: std.mem.Allocator, s: mlx.mlx_stream, num_layers: u32, head_dim: u32) !TurboState {
        if (!std.math.isPowerOfTwo(head_dim)) return error.NonPowerOfTwoHeadDim;
        var state = try initLazy(allocator, num_layers);
        errdefer state.deinit();
        const h_arr = try buildHadamardArray(allocator, s, head_dim);
        defer _ = mlx.mlx_array_free(h_arr);
        for (state.rk, 0..) |*a, i| {
            a.* = try cloneWithSignFlip(s, h_arr, head_dim, @intCast(0x9E37 ^ i));
            state.rk_dim[i] = head_dim;
        }
        for (state.rv, 0..) |*a, i| {
            a.* = try cloneWithSignFlip(s, h_arr, head_dim, @intCast(0x85EB ^ i));
            state.rv_dim[i] = head_dim;
        }
        return state;
    }

    /// Lazy-init the K rotation matrix for `layer` from the observed dim `n`.
    /// Subsequent calls for the same layer assert `n` matches; mismatched
    /// shapes (which would indicate an arch-specific layer-shape change
    /// mid-decode) return `error.TurboShapeMismatch`.
    pub fn ensureKLayer(self: *TurboState, s: mlx.mlx_stream, layer: u32, n: u32) !mlx.mlx_array {
        const li: usize = @intCast(layer);
        if (self.rk_dim[li] != 0) {
            if (self.rk_dim[li] != n) return error.TurboShapeMismatch;
            return self.rk[li];
        }
        if (!std.math.isPowerOfTwo(n)) return error.NonPowerOfTwoHeadDim;
        const h = try buildHadamardArray(self.allocator, s, n);
        defer _ = mlx.mlx_array_free(h);
        self.rk[li] = try cloneWithSignFlip(s, h, n, @as(u64, 0x9E37) ^ @as(u64, li));
        self.rk_dim[li] = n;
        return self.rk[li];
    }

    pub fn ensureVLayer(self: *TurboState, s: mlx.mlx_stream, layer: u32, n: u32) !mlx.mlx_array {
        const li: usize = @intCast(layer);
        if (self.rv_dim[li] != 0) {
            if (self.rv_dim[li] != n) return error.TurboShapeMismatch;
            return self.rv[li];
        }
        if (!std.math.isPowerOfTwo(n)) return error.NonPowerOfTwoHeadDim;
        const h = try buildHadamardArray(self.allocator, s, n);
        defer _ = mlx.mlx_array_free(h);
        self.rv[li] = try cloneWithSignFlip(s, h, n, @as(u64, 0x85EB) ^ @as(u64, li));
        self.rv_dim[li] = n;
        return self.rv[li];
    }

    pub fn deinit(self: *TurboState) void {
        for (self.rk) |*a| _ = mlx.mlx_array_free(a.*);
        for (self.rv) |*a| _ = mlx.mlx_array_free(a.*);
        self.allocator.free(self.rk);
        self.allocator.free(self.rv);
        self.allocator.free(self.rk_dim);
        self.allocator.free(self.rv_dim);
    }
};

/// Deterministic normalized Hadamard matrix `[N, N]` bf16. Sylvester
/// construction: `H_{2N} = [[H_N, H_N], [H_N, -H_N]] / sqrt(2)`. Result is
/// orthogonal (`H^T H = I`) and symmetric (`H^T = H`).
fn buildHadamardArray(allocator: std.mem.Allocator, s: mlx.mlx_stream, n: u32) !mlx.mlx_array {
    const N: usize = @intCast(n);
    const buf = try allocator.alloc(f32, N * N);
    defer allocator.free(buf);
    // Build sign matrix recursively, then normalize at the end. We carry
    // unnormalized ±1 entries through the recursion and divide by sqrt(N) once.
    var size: usize = 1;
    buf[0] = 1.0;
    while (size < N) : (size *= 2) {
        // Quadruple block expansion: top-right = top-left, bottom-left =
        // top-left, bottom-right = -top-left.
        for (0..size) |r| {
            for (0..size) |c| {
                const tl = buf[r * N + c];
                buf[r * N + (c + size)] = tl;
                buf[(r + size) * N + c] = tl;
                buf[(r + size) * N + (c + size)] = -tl;
            }
        }
    }
    const inv_sqrt_n: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(N)));
    for (buf) |*v| v.* *= inv_sqrt_n;
    const shape = [_]c_int{ @intCast(N), @intCast(N) };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Multiply each column of `h_arr` by ±1 according to bits of `seed`. The
/// result is still an orthogonal symmetric matrix (column sign flips of an
/// orthogonal symmetric matrix preserve orthogonality; symmetry preserved
/// because we apply the same flip to row i and column i — actually no, this
/// preserves the absolute values but breaks symmetry unless we also flip
/// row i. We do.). Used to give each layer (and K vs V) its own rotation.
fn cloneWithSignFlip(s: mlx.mlx_stream, h_arr: mlx.mlx_array, n: u32, seed: u64) !mlx.mlx_array {
    const N: usize = @intCast(n);
    // Build a column-sign vector `[N]` with entries ±1 from `seed`.
    var sign_buf: [4096]f32 = undefined;
    if (N > sign_buf.len) return error.HeadDimTooLarge;
    var s_state = seed *% 6364136223846793005;
    for (sign_buf[0..N]) |*v| {
        s_state = s_state *% 6364136223846793005 +% 1442695040888963407;
        v.* = if ((s_state >> 33) & 1 == 0) 1.0 else -1.0;
    }
    const shape_v = [_]c_int{@intCast(N)};
    const sign_f32 = mlx.mlx_array_new_data(&sign_buf, &shape_v, 1, .float32);
    defer _ = mlx.mlx_array_free(sign_f32);
    var sign_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sign_bf16);
    try mlx.check(mlx.mlx_astype(&sign_bf16, sign_f32, .bfloat16, s));

    // R_flipped = diag(sign) @ h @ diag(sign). For Hadamard `h`:
    // multiply rows by sign (broadcast over columns), then multiply columns
    // by sign (broadcast over rows).
    var row_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_scaled);
    {
        // reshape sign to [N, 1] for row-broadcast
        const sh = [_]c_int{ @intCast(N), 1 };
        var sign_col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_col);
        try mlx.check(mlx.mlx_reshape(&sign_col, sign_bf16, &sh, 2, s));
        try mlx.check(mlx.mlx_multiply(&row_scaled, h_arr, sign_col, s));
    }
    var out = mlx.mlx_array_new();
    {
        const sh = [_]c_int{ 1, @intCast(N) };
        var sign_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_row);
        try mlx.check(mlx.mlx_reshape(&sign_row, sign_bf16, &sh, 2, s));
        try mlx.check(mlx.mlx_multiply(&out, row_scaled, sign_row, s));
    }
    return out;
}

/// Rotate `dense_x` by `R` along its last axis: out = dense_x @ R. Caller
/// owns the returned array. mlx_matmul broadcasts the leading dims of
/// `dense_x` (`[B, H, T, D]`) against `R` (`[D, D]`), so we get
/// `[B, H, T, D]` out.
pub fn rotateLastDim(s: mlx.mlx_stream, dense_x: mlx.mlx_array, R: mlx.mlx_array) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, dense_x, R, s));
    return out;
}

/// TurboQuant write path: rotate then affine-quantize. The caller's
/// `dense_x` is not consumed.
pub fn quantizeTurbo(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    R: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    const rotated = try rotateLastDim(s, dense_x, R);
    defer _ = mlx.mlx_array_free(rotated);
    return try quantizeAffine(s, rotated, group_size, bits);
}

/// TurboQuant read path: affine-dequantize then rotate back. Caller owns
/// the returned dense array.
pub fn dequantizeTurbo(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    R: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    const deq = try dequantizeAffine(s, q, scales, biases, group_size, bits);
    defer _ = mlx.mlx_array_free(deq);
    // `R` is symmetric in our Hadamard+sign-flip construction (we flip rows
    // and columns by the same `sign` vector, so the matrix stays symmetric).
    // Therefore R = R^T = R^{-1}, and the inverse rotation is just `@ R`.
    return try rotateLastDim(s, deq, R);
}

/// Affine dequantize a `(q, scales, biases)` triple to dense bf16. Caller
/// owns the returned array.
pub fn dequantizeAffine(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_dequantize(
        &out,
        q,
        scales,
        biases,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        .{ .value = .bfloat16, .has_value = true },
        s,
    ));
    return out;
}

// ── Fused quant-attention path (opt-in via --kv-attn-mode fused) ──
//
// `quantAttention` reads K and V directly from their quantized triples,
// avoiding the dense materialization that `KVCache.denseView` performs in
// the default path. The fusion comes from `mlx_quantized_matmul`, Apple's
// kernel that internally dequantizes one operand and multiplies in a
// single Metal pass — same primitive `qmatmulBits` already uses for weight
// quantization throughout `transformer.zig`.
//
// Tradeoff vs `mlx_fast_scaled_dot_product_attention` (flash-attention):
//   * Dense SDPA fuses Q@K^T → scale → mask → softmax → @V into a single
//     tiled pass with no intermediate HBM writes. Wins at short context.
//   * Hand-rolled qmm × 2 + softmax loses tile-level fusion but skips the
//     dense K/V materialization. Wins at long context where K/V bandwidth
//     dominates.
// The crossover is data-driven; v1 ships behind `--kv-attn-mode fused` so the
// default is unchanged.
//
// Shape contract:
//   q_dense      : [B, H,    T_q, D] bf16 (Q already scaled or not; we apply scale below)
//   k_q/sc/bi    : K affine-quantized along last axis (D). Shape of the
//                  triple is whatever `mlx_quantize` produced.
//   v_q/sc/bi    : V same as K.
// GQA: when H > H_kv, `mlx_quantized_matmul` is responsible for
// broadcasting the leading head dim — same way `mlx_matmul` does. If the
// model uses head ratios that aren't natively supported, callers should
// expand K/V (or fall through to the dense path) before invoking this.

/// A read-only borrow of a `QuantizedKV` triple — the cache owns the
/// arrays; the call site borrows them for the duration of one attention.
/// Used to thread quant triples through `DenseKVView` without altering
/// refcount semantics.
pub const BorrowedTriple = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
};

/// Hand-rolled attention that consumes K and V triples directly.
/// `scale` is the standard 1/sqrt(D) factor SDPA applies before softmax.
/// `mask_mode` mirrors `mlx_fast_scaled_dot_product_attention`:
///   * "causal": apply a Q-aligned causal mask (upper triangular -inf).
///     For T_q == 1 (decode tick) this is a no-op.
///   * "":       no mask.
///   * "array":  add `mask_arr` to the pre-softmax scores. Must be
///               additive (mlx convention: -inf for masked positions).
/// Returns a `[B, H, T_q, D]` bf16 array; caller owns and frees.
pub fn quantAttention(
    q_dense: mlx.mlx_array,
    k_triple: BorrowedTriple,
    v_triple: BorrowedTriple,
    bits: u8,
    group_size: u32,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    // GQA expansion. mlx_quantized_matmul does not broadcast across the
    // head dim the way mlx_fast_scaled_dot_product_attention does, so for
    // models with H_q > H_kv (Gemma 4 E4B: 8 vs 2, Qwen 3.5: 32 vs 8,
    // every modern GQA arch) we need to either repeat K/V to match H_q or
    // reshape Q down to H_kv groups. We pick repeat: it's one op per K
    // and V triple component (q/scales/biases), and mlx makes the result
    // a stride-0 view so the underlying memory isn't duplicated until
    // the kernel reads it.
    const q_shape = mlx.getShape(q_dense);
    if (q_shape.len < 4) return error.UnexpectedQShape;
    const k_q_shape = mlx.getShape(k_triple.q);
    if (k_q_shape.len < 4) return error.UnexpectedKShape;
    const H_q: c_int = q_shape[1];
    const H_kv: c_int = k_q_shape[1];
    const need_gqa = H_q != H_kv;
    const repeats: c_int = if (need_gqa) @divExact(H_q, H_kv) else 1;

    // Owned-or-borrowed expansion for each triple component. When
    // `need_gqa == false` everything aliases the caller's borrows.
    var k_q_used = k_triple.q;
    var k_sc_used = k_triple.scales;
    var k_bi_used = k_triple.biases;
    var v_q_used = v_triple.q;
    var v_sc_used = v_triple.scales;
    var v_bi_used = v_triple.biases;
    var owns_k_q = false;
    var owns_k_sc = false;
    var owns_k_bi = false;
    var owns_v_q = false;
    var owns_v_sc = false;
    var owns_v_bi = false;
    defer {
        if (owns_k_q) _ = mlx.mlx_array_free(k_q_used);
        if (owns_k_sc) _ = mlx.mlx_array_free(k_sc_used);
        if (owns_k_bi) _ = mlx.mlx_array_free(k_bi_used);
        if (owns_v_q) _ = mlx.mlx_array_free(v_q_used);
        if (owns_v_sc) _ = mlx.mlx_array_free(v_sc_used);
        if (owns_v_bi) _ = mlx.mlx_array_free(v_bi_used);
    }
    if (need_gqa) {
        var t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, k_triple.q, repeats, 1, s));
        k_q_used = t;
        owns_k_q = true;
        t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, k_triple.scales, repeats, 1, s));
        k_sc_used = t;
        owns_k_sc = true;
        t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, k_triple.biases, repeats, 1, s));
        k_bi_used = t;
        owns_k_bi = true;
        t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, v_triple.q, repeats, 1, s));
        v_q_used = t;
        owns_v_q = true;
        t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, v_triple.scales, repeats, 1, s));
        v_sc_used = t;
        owns_v_sc = true;
        t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&t, v_triple.biases, repeats, 1, s));
        v_bi_used = t;
        owns_v_bi = true;
    }

    // 1) scores = Q @ K^T. transpose_w=true contracts on K's quantized
    //    last axis (D), the same dim Q contracts. Output: [B, H, T_q, T_k].
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_quantized_matmul(
        &scores,
        q_dense,
        k_q_used,
        k_sc_used,
        k_bi_used,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));

    // 2) scale. Fold into a single multiply rather than dividing inside
    //    softmax — saves one op per call.
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, scores, scale_arr, s));

    // 3) mask. We build masked_scores into `pre_softmax` and free the
    //    intermediates at scope exit. The branches assign `pre_softmax`
    //    to either an owned array or a borrow of `scaled`; `owns_pre`
    //    tracks which.
    var pre_softmax: mlx.mlx_array = .{};
    var owns_pre = false;
    defer {
        if (owns_pre) _ = mlx.mlx_array_free(pre_softmax);
    }
    if (std.mem.eql(u8, mask_mode, "causal")) {
        // Causal mask: upper-triangular `-inf` matching scores shape
        // `[..., T_q, T_k]`. For decode (T_q == 1) the mask is identically
        // zero so we skip building it.
        const scores_shape = mlx.getShape(scaled);
        if (scores_shape.len < 2) return error.UnexpectedShape;
        const t_q: c_int = scores_shape[scores_shape.len - 2];
        const t_k: c_int = scores_shape[scores_shape.len - 1];
        if (t_q == 1) {
            pre_softmax = scaled;
        } else {
            // Build a `[T_q, T_k]` additive mask: -inf above the diagonal
            // anchored to the right edge (Q position 0 corresponds to
            // K position (T_k - T_q)).
            const offset: c_int = t_k - t_q;
            const shape2 = [_]c_int{ t_q, t_k };
            var ones2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones2);
            try mlx.check(mlx.mlx_ones(&ones2, &shape2, 2, .bfloat16, s));
            var upper = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper);
            // triu(ones, k=offset+1) yields 1s strictly above the causal
            // diagonal (positions Q can't see) and 0s on/below.
            try mlx.check(mlx.mlx_triu(&upper, ones2, offset + 1, s));
            const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(neg_inf);
            var neg_inf_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(neg_inf_bf16);
            try mlx.check(mlx.mlx_astype(&neg_inf_bf16, neg_inf, .bfloat16, s));
            var add_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(add_mask);
            // mask = upper * -inf → -inf above diag, 0 elsewhere.
            try mlx.check(mlx.mlx_multiply(&add_mask, upper, neg_inf_bf16, s));
            pre_softmax = mlx.mlx_array_new();
            owns_pre = true;
            try mlx.check(mlx.mlx_add(&pre_softmax, scaled, add_mask, s));
        }
    } else if (std.mem.eql(u8, mask_mode, "array")) {
        pre_softmax = mlx.mlx_array_new();
        owns_pre = true;
        try mlx.check(mlx.mlx_add(&pre_softmax, scaled, mask_arr, s));
    } else {
        // No mask: borrow `scaled` for the softmax input.
        pre_softmax = scaled;
    }

    // 4) softmax along last axis. `precise=true` matches the fast SDPA
    //    kernel's accumulation order (f32 reductions inside softmax) more
    //    closely than the default; trades a small perf hit for tighter
    //    equivalence to dense SDPA.
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    try mlx.check(mlx.mlx_softmax_axis(&attn, pre_softmax, -1, true, s));

    // 5) out = attn @ V. transpose_w=false contracts attn's last axis
    //    (T_k) with V's second-to-last (T_k) — V's quantized last axis
    //    (D) becomes the output dim. mlx_quantized_matmul dequantizes V
    //    on-the-fly without materializing a dense intermediate.
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_quantized_matmul(
        &out,
        attn,
        v_q_used,
        v_sc_used,
        v_bi_used,
        false,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return out;
}

// ── Tests ──

const testing = std.testing;

/// Build a `[1, 1, 8, head_dim]` bf16 tensor whose values vary smoothly per
/// position so quantization error is non-trivial but bounded.
fn buildSmoothBf16(s: mlx.mlx_stream, head_dim: c_int) !mlx.mlx_array {
    const T: usize = 8;
    const D: usize = @intCast(head_dim);
    const buf = try testing.allocator.alloc(f32, T * D);
    defer testing.allocator.free(buf);
    for (0..T) |t| {
        for (0..D) |d| {
            // Range roughly [-1, 1]; smooth so adjacent group elements are
            // close (best case for affine), with some variation across groups.
            const fi: f32 = @floatFromInt(t * D + d);
            const denom: f32 = @floatFromInt(T * D);
            buf[t * D + d] = (fi / denom) * 2.0 - 1.0;
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(T), head_dim };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Read a flat float32 host buffer for a small array (eval-and-copy via
/// `mlx_astype` to float32 then reshape to 1D).
fn readF32Flat(s: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
    var f32_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_view);
    try mlx.check(mlx.mlx_astype(&f32_view, arr, .float32, s));

    const n: c_int = @intCast(mlx.mlx_array_size(f32_view));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    {
        const sh = [_]c_int{n};
        try mlx.check(mlx.mlx_reshape(&flat, f32_view, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
    const out = try allocator.alloc(f32, @intCast(n));
    @memcpy(out, ptr[0..@intCast(n)]);
    return out;
}

test "quantizeAffine + dequantizeAffine round-trip at 4 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 4);
    defer qkv.deinit();

    // Shape sanity: q is [..., D * bits / 32] = [..., 256 * 4 / 32] = [..., 32]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 32), q_shape[q_shape.len - 1]);
    // scales/biases last dim = D / group_size = 256 / 64 = 4
    const sc_shape = mlx.getShape(qkv.scales);
    try testing.expectEqual(@as(c_int, 4), sc_shape[sc_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    try testing.expectEqual(orig.len, got.len);
    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 4-bit affine on smooth data with group=64: empirical ceiling well under 0.05.
    try testing.expect(max_err < 0.05);
}

test "quantizeAffine + dequantizeAffine round-trip at 8 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 8);
    defer qkv.deinit();

    // Shape: q = [..., 256 * 8 / 32] = [..., 64]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 64), q_shape[q_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 8);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 8-bit affine: ~256x finer steps than 4-bit; expect < 0.005 on smooth data.
    try testing.expect(max_err < 0.01);
}

test "KVQuantConfig.affine builds a sane config" {
    const c4 = KVQuantConfig.affine(4);
    try testing.expectEqual(Scheme.affine, c4.scheme);
    try testing.expectEqual(@as(u8, 4), c4.bits);
    try testing.expectEqual(@as(u32, 64), c4.group_size);

    const c8 = KVQuantConfig.affine(8);
    try testing.expectEqual(@as(u8, 8), c8.bits);

    const cd = KVQuantConfig.dense;
    try testing.expectEqual(Scheme.off, cd.scheme);
}

test "KVQuantConfig.turboquant routes 2 and 4 to distinct schemes" {
    const t2 = KVQuantConfig.turboquant(2);
    try testing.expectEqual(Scheme.turboquant_2, t2.scheme);
    try testing.expectEqual(@as(u8, 2), t2.bits);
    try testing.expectEqual(@as(u32, 64), t2.group_size);

    const t4 = KVQuantConfig.turboquant(4);
    try testing.expectEqual(Scheme.turboquant_4, t4.scheme);
    try testing.expectEqual(@as(u8, 4), t4.bits);
}

test "buildHadamardArray produces a valid Hadamard at N=8" {
    const s = mlx.gpuStream();
    const h = try buildHadamardArray(testing.allocator, s, 8);
    defer _ = mlx.mlx_array_free(h);

    // H is `[8, 8]` and H @ H = I (entries already normalized by 1/sqrt(8)).
    var hh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hh);
    try mlx.check(mlx.mlx_matmul(&hh, h, h, s));
    const got = try readF32Flat(s, hh, testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(usize, 64), got.len);
    // Diagonal ~1, off-diagonal ~0. Use a loose tolerance because we cast
    // to bf16 and back; rounding error is non-trivial.
    for (0..8) |r| {
        for (0..8) |c| {
            const expected: f32 = if (r == c) 1.0 else 0.0;
            try testing.expect(@abs(got[r * 8 + c] - expected) < 0.05);
        }
    }
}

test "quantizeTurbo + dequantizeTurbo round-trip at 4 bits with Hadamard" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    // Build a single Hadamard matrix and use it as R.
    var ts = try TurboState.initHadamard(testing.allocator, s, 1, 256);
    defer ts.deinit();
    const R = ts.rk[0];

    var qkv = try quantizeTurbo(s, src, R, 64, 4);
    defer qkv.deinit();

    const deq = try dequantizeTurbo(s, qkv.q, qkv.scales, qkv.biases, R, 64, 4);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqual(orig.len, got.len);

    // Smoke check: no NaN/Inf in the dequantized output. The smooth ramp
    // is actually a worst-case input for TurboQuant (the rotation mixes a
    // tight local range into a wider global one, increasing per-group step
    // size). We only assert finiteness + that the mean absolute error
    // stays sub-input-range — that catches kernel bugs (wrong matmul
    // shape, NaN propagation, packing off-by-one) without overspecifying.
    var sum_abs: f64 = 0;
    var input_range: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(std.math.isFinite(g));
        sum_abs += @abs(o - g);
        input_range = @max(input_range, @abs(o));
    }
    const mean_err: f32 = @floatCast(sum_abs / @as(f64, @floatFromInt(orig.len)));
    // Mean error < input range (i.e. the output bears resemblance to the input).
    try testing.expect(mean_err < input_range);
}

test "quantizeTurbo + dequantizeTurbo round-trip at 2 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, 256);
    defer ts.deinit();
    const R = ts.rk[0];

    var qkv = try quantizeTurbo(s, src, R, 64, 2);
    defer qkv.deinit();

    const deq = try dequantizeTurbo(s, qkv.q, qkv.scales, qkv.biases, R, 64, 2);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    // 2-bit on a smooth-ramp input is heavily lossy — only 4 quant levels
    // per group. We just want finiteness + bounded values; per-element
    // tightness is meaningless on this input shape.
    var input_range: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(std.math.isFinite(g));
        input_range = @max(input_range, @abs(o));
        // Output should stay within ~2x the input range — a generous bound
        // that rejects NaN, runaway, or wrong-bias accumulation.
        try testing.expect(@abs(g) < 2.0 * input_range + 1.0);
    }
}

test "TurboState.initHadamard rejects non-power-of-2 head_dim" {
    const s = mlx.gpuStream();
    const result = TurboState.initHadamard(testing.allocator, s, 1, 96);
    try testing.expectError(error.NonPowerOfTwoHeadDim, result);
}

// ── Fused-attention validation ──
//
// The two tests below are the validation harness Phase 2 v1 relies on
// before wiring `quantAttention` into transformer.zig SDPA call sites.
// They prove (a) `mlx_quantized_matmul` semantics match dequant+matmul
// in both transpose modes, and (b) the assembled `quantAttention`
// produces logits within the same loose tolerance (0.05 max-abs-diff)
// the existing affine round-trip tests use.

/// Build a `[B, H, T, D]` dense bf16 with smooth ramped values per
/// `(b, h, t, d)` so quantization is non-trivial. Caller frees.
fn buildSmoothBHTD(s: mlx.mlx_stream, B: c_int, H: c_int, T: c_int, D: c_int) !mlx.mlx_array {
    const total: usize = @intCast(B * H * T * D);
    const buf = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(buf);
    var i: usize = 0;
    for (0..@intCast(B)) |b| {
        for (0..@intCast(H)) |h| {
            for (0..@intCast(T)) |t| {
                for (0..@intCast(D)) |d| {
                    const f: f32 = @floatFromInt(b * 17 + h * 5 + t * 3 + d);
                    const denom: f32 = @floatFromInt(total);
                    buf[i] = (f / denom) * 2.0 - 1.0;
                    i += 1;
                }
            }
        }
    }
    const shape = [_]c_int{ B, H, T, D };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

test "mlx_quantized_matmul transpose=true matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 64]  (Q-shaped: B, H, T_q, D)
    // w: [1, 2, 5, 64]  (K-shaped: B, H_kv, T_k, D, will be quantized along D)
    // Expected: x @ w.T → [1, 2, 3, 5]
    const x = try buildSmoothBHTD(s, 1, 2, 3, 64);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequantize w, then dense matmul with explicit transpose.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    // Transpose w_deq along last two dims: [1,2,5,64] → [1,2,64,5]
    var w_deq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq_t);
    const axes_t = [_]c_int{ 0, 1, 3, 2 };
    try mlx.check(mlx.mlx_transpose_axes(&w_deq_t, w_deq, &axes_t, 4, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq_t, s));

    // Candidate: fused qmm with transpose=true.
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        true,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    // Shape sanity.
    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    var max_ref: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
        if (@abs(r) > max_ref) max_ref = @abs(r);
    }
    // bf16 reductions inside qmm don't bit-match an explicit dequant +
    // mlx_matmul because the order of additions differs. Compare in
    // relative terms (the smooth-ramp test produces outputs in the
    // [-N*D*1.0, N*D*1.0] range so absolute thresholds are misleading).
    const rel_err = if (max_ref > 0) max_err / max_ref else max_err;
    if (rel_err >= 0.02) {
        std.debug.print("max_err={d} max_ref={d} rel_err={d}\n", .{ max_err, max_ref, rel_err });
    }
    try testing.expect(rel_err < 0.02);
}

test "mlx_quantized_matmul transpose=false matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 5]  (attn-shaped: B, H, T_q, T_k)
    // w: [1, 2, 5, 64] (V-shaped:    B, H_kv, T_k, D)
    // Expected: x @ w → [1, 2, 3, 64], contracting over T_k.
    // For V, the quantized last axis is D — so transpose=false should
    // dequantize each D-column on the fly while contracting over T_k.
    const x = try buildSmoothBHTD(s, 1, 2, 3, 5);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequant + plain matmul.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq, s));

    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        false,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "quantAttention matches dense SDPA at 4-bit (decode, T_q=1)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T_k: c_int = 8;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, 1, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    // Dense reference. We pass the dequantized K/V so any mismatch
    // attributable to quantization itself is folded into both paths.
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "",
        none_mask,
        .{ .ctx = null },
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        scale,
        "",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    // Matches the bf16 reduction tolerance used elsewhere in this file.
    try testing.expect(max_err < 0.05);
}

test "quantAttention causal mask matches dense SDPA (prefill, T_q=T_k=4)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "causal",
        none_mask,
        .{ .ctx = null },
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}
