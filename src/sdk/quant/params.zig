//! sdk/quant/params.
//! Owns BITS_CACHE_CAP, QuantParams, QuantParamsCache, BitsCache, quantParamsOrDense, affineParamsFromGeometry, and 4 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;

// ── Quantization params cache ──
// A tiny lock-free pointer → (bits, group_size) cache. Quantized weights are loaded
// once at init and reused for every forward pass; we detect (or pre-bind) params on
// first touch and serve hits thereafter. Uses open addressing with linear probing on
// a fixed-size array — this fits in L1 and keeps the cost of a lookup to ~5ns,
// matching the perf commit's intent of "eliminate per-call detect overhead" while
// supporting mixed-precision quantization (Gemma-4 MoE per-layer bits, etc.).
pub const BITS_CACHE_CAP: usize = 1024; // plenty for 60 layers × ~10 quant weights × factor
pub const QuantParams = struct {
    bits: u32,
    group_size: u32,
    mode: QuantMode = .affine,

    /// The model-wide scheme a config declares (`quant_bits`, `quant_group_size`, `quant_mode`).
    pub fn of(cfg: anytype) QuantParams {
        return .{ .bits = cfg.quant_bits, .group_size = cfg.quant_group_size, .mode = cfg.quant_mode };
    }
};

const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = @splat(null),
    vals_bits: [BITS_CACHE_CAP]u8 = @splat(0),
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = @splat(0),
    vals_mode: [BITS_CACHE_CAP]u8 = @splat(0),

    pub inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert params for `key`. Returns false if the probe window saturated
    /// (caller should treat as "fall through to detection" — but realistically
    /// never fires given BITS_CACHE_CAP is 1024 vs ~2k weights max).
    pub fn put(self: *QuantParamsCache, key: *anyopaque, qp: QuantParams) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(qp.bits);
                self.vals_gs_div8[idx] = @intCast(qp.group_size / 8);
                self.vals_mode[idx] = @backingInt(qp.mode);
                return true;
            }
        }
        return false;
    }
};

// Backwards-compatibility alias — keeps existing field names readable.
pub const BitsCache = QuantParamsCache;

/// `computeQuantParams` for a weight that may be dense (null-ctx scales), as
/// `quantParamsHinted` does for the Transformer methods.
pub fn quantParamsOrDense(model: QuantParams, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    if (sc.ctx == null) return model;
    return computeQuantParams(model, w, sc, in_dim);
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

/// The fp mode a solved (bits, group_size) pair belongs to. Each format
/// DEFINES its block size — MXFP8/MXFP4 are 32, NVFP4 is 16 — so the pair
/// names exactly one of them. Anything else is a layout we don't serve;
/// returning null keeps a checkpoint's own declaration rather than inventing a
/// mode for it.
fn fpQuantModeFor(bits: u32, group_size: u32) ?QuantMode {
    if (bits == 8 and group_size == 32) return .mxfp8;
    if (bits == 4 and group_size == 32) return .mxfp4;
    if (bits == 4 and group_size == 16) return .nvfp4;
    return null;
}

/// Per-weight (bits, group_size, mode) for an fp-quantized tensor, solved from
/// the packed geometry against the activation's inner dim: a row holds
/// `w_cols * 32` packed bits over `in_dim` values, and `s_cols` scales cover
/// them in groups.
///
/// Returns null when the geometry AGREES with the model-wide declaration —
/// the caller then returns the config verbatim, so a consistent checkpoint
/// resolves exactly as it did before this existed.
fn fpParamsFromGeometry(model: QuantParams, w_cols: u32, s_cols: u32, in_dim: u32) ?QuantParams {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = w_cols * 32;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const group_size = in_dim / s_cols;
    if (bits == model.bits and group_size == model.group_size) return null;
    const mode = fpQuantModeFor(bits, group_size) orelse return null;
    return .{ .bits = bits, .group_size = group_size, .mode = mode };
}

/// Per-weight (bits, group_size, mode) solved from packed geometry ALONE, for
/// callers that have no model-wide scheme to consult — an MTP sidecar file carries
/// weights and nothing else. The mode is separated on the same evidence
/// `computeQuantParams` uses: affine is the only mode whose checkpoints ship
/// per-group biases, and every fp mode stores fp8-encoded `uint8` scales, so
/// bias-less + uint8 scales resolves through the (bits, group_size) table.
pub fn quantParamsFromGeometry(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    biases_present: bool,
    in_dim: u32,
) ?QuantParams {
    if (sc.ctx == null or in_dim == 0) return null;
    if (biases_present or mlx.mlx_array_dtype(sc) != .uint8) {
        return affineParamsFromGeometry(w, sc, in_dim);
    }
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (w_cols == 0 or s_cols == 0) return null;
    if ((w_cols * 32) % in_dim != 0 or in_dim % s_cols != 0) return null;
    const mode = fpQuantModeFor((w_cols * 32) / in_dim, in_dim / s_cols) orelse return null;
    return .{ .bits = (w_cols * 32) / in_dim, .group_size = in_dim / s_cols, .mode = mode };
}

pub fn computeQuantParams(model: QuantParams, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;

    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (model.mode != .affine) {
            // A per-tensor override can live INSIDE the fp family: LFM2.5-2.6B-MLX's
            // mxfp4 build stores `model.embed_tokens` as mxfp8. Every fp mode
            // carries uint8 (E8M0/E4M3) scales, so the scales-dtype rule above
            // cannot separate them and the embedding resolved at the model-wide
            // 4 bits — an `mlx_dequantize` shape rejection, i.e. an MLX error,
            // i.e. a dead server on the first request. The packed geometry says
            // it exactly; solve, exactly as the affine arm below does.
            if (in_dim) |in| {
                if (fpParamsFromGeometry(model, w_cols, s_cols, in)) |qp| return qp;
            } else if (s_cols > 0 and (w_cols * 32) % s_cols == 0) {
                const per_group = (w_cols * 32) / s_cols;
                const bits: u32 = if (per_group == 256) 8 else 4;
                const gs = per_group / bits;
                if (fpQuantModeFor(bits, gs)) |mode| return .{ .bits = bits, .group_size = gs, .mode = mode };
            }
            return model;
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }

    if (model.mode == .affine) {
        // Exact per-weight solve when the caller knows the input dim — a
        // hinted tensor whose geometry contradicts the config's group size
        // (MTP sidecars mix gs 128/64/32 over a gs-64 trunk) resolves to its
        // TRUE params. Consistent tensors solve to the config values anyway.
        if (in_dim) |in| {
            if (affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        }
        return .{
            .bits = detectQuantBits(w, sc, model.group_size),
            .group_size = model.group_size,
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

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,
    /// Raw ggml blocks (src/sdk/quant/gguf). Never reaches an MLX quantized op:
    /// the per-tensor type rides on the weight, see `mlx_gguf.kernels.Info`.
    gguf,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
            .gguf => "gguf",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }

    /// The e2m1 nibble modes (byte scales, no biases): MXFP4 g32, NVFP4 g16.
    pub fn isFp4(self: QuantMode) bool {
        return self == .mxfp4 or self == .nvfp4;
    }
};

/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
pub fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

// ── tests ──

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
    const cfg: QuantParams = .{ .bits = 4, .group_size = 64, .mode = .affine };
    const qp_via = computeQuantParams(cfg, wq, sq, H);
    try testing.expectEqual(@as(u32, 5), qp_via.bits);
    try testing.expectEqual(@as(u32, 128), qp_via.group_size);
}

test "computeQuantParams resolves an fp override INSIDE the fp family (mxfp8 embed under mxfp4)" {
    // LiquidAI/LFM2.5-2.6B-MLX/mxfp4 declares a model-wide mxfp4/gs32 AND a
    // per-tensor override: `"model.embed_tokens": {bits 8, group_size 32,
    // mode "mxfp8"}`. Both modes carry uint8 (E8M0) scales, so the
    // scales-dtype rule cannot tell them apart and the embedding resolved at
    // the model-wide 4 bits — `mlx_dequantize` then rejected the shapes
    // ("matrix of shape (1,512) and scales of shape (1,64)"), and an MLX error
    // is not a Zig error: the first request killed the server.
    //
    // The packed geometry states the answer exactly, the same way the affine
    // path already solves off-config sidecar quants.
    const s = mlx.gpuStream();
    const H = 2048; // lfm2 hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    const cfg: QuantParams = .{ .bits = 4, .group_size = 32, .mode = .mxfp4 };

    // embed_tokens: w [V, 512] u32 + scales [V, 64] u8 → 2048 values/row at
    // 8 bits, gs 32. (Real shapes; V shrunk — only the last axis is read.)
    const we = try mk.arr(&.{ 1024, 512 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 1024, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = computeQuantParams(cfg, we, se, H);
    try testing.expectEqual(QuantMode.mxfp8, qp_e.mode);
    try testing.expectEqual(@as(u32, 8), qp_e.bits);
    try testing.expectEqual(@as(u32, 32), qp_e.group_size);

    // An ordinary mxfp4 linear in the SAME model: w [2048, 256] + [2048, 64].
    // Unchanged — the config already describes it.
    const wl = try mk.arr(&.{ 2048, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wl);
    const sl = try mk.arr(&.{ 2048, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(sl);
    const qp_l = computeQuantParams(cfg, wl, sl, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_l.mode);
    try testing.expectEqual(@as(u32, 4), qp_l.bits);
    try testing.expectEqual(@as(u32, 32), qp_l.group_size);

    // An nvfp4 model's own tensors (gs 16) stay nvfp4 — the solve agrees with
    // the config, so it never re-labels a consistent checkpoint.
    const cfg_nv: QuantParams = .{ .bits = 4, .group_size = 16, .mode = .nvfp4 };
    const s_nv = try mk.arr(&.{ 2048, 128 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s_nv);
    const qp_nv = computeQuantParams(cfg_nv, wl, s_nv, H);
    try testing.expectEqual(QuantMode.nvfp4, qp_nv.mode);
    try testing.expectEqual(@as(u32, 16), qp_nv.group_size);

    // A geometry that maps to no fp format we serve is NOT invented into one:
    // the config stands, and the load fails honestly if it really disagrees.
    const s_odd = try mk.arr(&.{ 2048, 21 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s_odd);
    const qp_odd = computeQuantParams(cfg, wl, s_odd, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_odd.mode);
    try testing.expectEqual(@as(u32, 4), qp_odd.bits);
    try testing.expectEqual(@as(u32, 32), qp_odd.group_size);

    // No inner-dim hint: bits x group_size (256 / 128 / 64) still names one fp format.
    const qp_nohint = computeQuantParams(cfg, we, se, null);
    try testing.expectEqual(QuantMode.mxfp8, qp_nohint.mode);
    try testing.expectEqual(@as(u32, 8), qp_nohint.bits);
    const qp_nohint_nv = computeQuantParams(cfg, wl, s_nv, null);
    try testing.expectEqual(QuantMode.nvfp4, qp_nohint_nv.mode);
    try testing.expectEqual(@as(u32, 16), qp_nohint_nv.group_size);
    const qp_nohint_odd = computeQuantParams(cfg, wl, s_odd, null);
    try testing.expectEqual(QuantMode.mxfp4, qp_nohint_odd.mode);
}

test "computeQuantParams resolves mixed nvfp4 + affine-override weights" {
    // Real-world shape: gemma-4 QAT nvfp4 checkpoints quantize most tensors
    // nvfp4 (uint8 fp8 scales, gs 16) but override the shared MLP to affine
    // 8-bit/gs64 (bf16 scales + biases). Detection key: scales dtype, then
    // the activation inner dim to pin (bits, group_size) for the override.
    const s = mlx.gpuStream();
    const IN = 2560;
    const OUT = 8;

    const cfg: QuantParams = .{ .bits = 4, .group_size = 16, .mode = .nvfp4 };

    // nvfp4 tensor: w [OUT, IN*4/32] u32, scales [OUT, IN/16] u8.
    var w_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_nv);
    var sc_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_nv);
    const w_nv_sh = [_]c_int{ OUT, IN * 4 / 32 };
    const sc_nv_sh = [_]c_int{ OUT, IN / 16 };
    try mlx.check(mlx.mlx_zeros(&w_nv, &w_nv_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_nv, &sc_nv_sh, 2, .uint8, s));
    const qp_nv = computeQuantParams(cfg, w_nv, sc_nv, null);
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
    const qp_af = computeQuantParams(cfg, w_af, sc_af, IN);
    try testing.expectEqual(QuantMode.affine, qp_af.mode);
    try testing.expectEqual(@as(u32, 8), qp_af.bits);
    try testing.expectEqual(@as(u32, 64), qp_af.group_size);

    // Without a hint: falls back to mlx-lm's override default gs=64.
    const qp_af_nohint = computeQuantParams(cfg, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_af_nohint.mode);
    try testing.expectEqual(@as(u32, 8), qp_af_nohint.bits);
    try testing.expectEqual(@as(u32, 64), qp_af_nohint.group_size);

    // Plain affine model (config mode affine, bf16 scales): unchanged
    // detect-against-config-gs behavior.
    const cfg_affine: QuantParams = .{ .bits = 4, .group_size = 64, .mode = .affine };
    const qp_plain = computeQuantParams(cfg_affine, w_af, sc_af, null);
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
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
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
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}
