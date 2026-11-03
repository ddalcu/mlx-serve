//! The legacy `Transformer`'s methods over `sdk/blocks/linear/qmatmul.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const qmv2 = @import("../../../sdk/blocks/qmm/qmv2.zig");
const mlx_gguf = @import("../../../sdk/quant/gguf/root.zig");
const sushi_pack = @import("../../../sdk/quant/exl3/sushi_pack.zig");
const qmv_nax2 = @import("../../../sdk/blocks/qmm/qmv_nax2.zig");
const qmm_int8 = @import("../../../sdk/blocks/qmm/qmm_int8.zig");
const lastDim = @import("../../../mlx/array.zig").lastDim;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const Transformer = @import("../transformer.zig").Transformer;
const BITS_CACHE_CAP = @import("../../../sdk/quant/params.zig").BITS_CACHE_CAP;
const BitsCache = @import("../../../sdk/quant/params.zig").BitsCache;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const computeQuantParams = @import("../../../sdk/quant/params.zig").computeQuantParams;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const rowExactQmm = @import("../../../sdk/blocks/linear/qmatmul.zig").rowExactQmm;

pub inline fn qmatmul(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    if (mlx_gguf.kernels.infoOf(w, sc)) |info| return mlx_gguf.kernels.linear(info, x, w, self.s);
    if (sushi_pack.isRowScaledFp8(w, sc)) return sushi_pack.fp8Linear(x, w, sc, self.s);
    // Resolve (bits, group_size, mode) per weight. Most weights inherit the
    // global config; per-weight overrides (mixed-precision checkpoints, e.g.
    // affine 8-bit shared MLP inside an nvfp4 QAT model) are detected on
    // first touch — x's inner dim pins (bits, group_size) exactly.
    const qp = self.quantParamsHinted(w, sc, lastDim(x));
    // The 2-bit kernels (qmv2.zig) serve Prism packs: Hadamard ones, and
    // ternary ones without rotations. Every other 2-bit pack keeps stock qmm.
    if (self.rht) |reg| {
        const signs = reg.get(w);
        // Prompt width with int8 on: the rotation rides the quantizer, so the
        // rotated activation is never written (the narrow routes stop at 16 rows).
        if (self.int8_cache) |ic| if (signs) |sg| if (try qmm_int8.qmmRotated(ic, x, sg, reg.block, w, sc, bi, qp.bits, qp.group_size, reg.bias_is_neg_scale, self.s)) |y| return y;
        const xr = if (signs) |sg| try reg.applyIn(x, sg, self.s) else x;
        defer if (signs != null) {
            _ = mlx.mlx_array_free(xr);
        };
        // nax2 first: it declines below its measured crossover, and from
        // 6 rows up it beats qmv2 on every 27B projection shape.
        if (try qmv_nax2.qmm(xr, w, sc, bi, qp.bits, qp.group_size, self.s)) |y| return y;
        if (try qmv2.qmm(xr, w, sc, bi, qp.bits, qp.group_size, signs != null and reg.bias_is_neg_scale, true, self.s)) |y| return y;
        // OPT-IN and LOSSY (activations to int8); declines unless the
        // model or process enables it, and below prompt width.
        if (self.int8_cache) |ic| if (try qmm_int8.qmm(ic, xr, w, sc, bi, qp.bits, qp.group_size, signs != null and reg.bias_is_neg_scale, self.s)) |y| return y;
        return qmatmulBits(xr, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
    }
    if (self.ternary_2bit) {
        if (try qmv_nax2.qmm(x, w, sc, bi, qp.bits, qp.group_size, self.s)) |y| return y;
        if (try qmv2.qmm(x, w, sc, bi, qp.bits, qp.group_size, true, false, self.s)) |y| return y;
    }
    if (self.config.rowExactDecode() and qp.mode == .affine) {
        if (try rowExactQmm(std.mem.eql(u8, self.config.model_type, "nemotron_h"), x, w, sc, bi, qp.bits, qp.group_size, self.s)) |y| return y;
    }
    return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
}

/// Final logits projection. For dense bf16 the lm_head weight is [vocab, hidden]
/// and is NOT pre-transposed at load (when tied it aliases emb_w, which must stay
/// [vocab, hidden] for the embedding lookup), so we project via a lazy transposed
/// view. Quantized models fall through to the standard gather/qmm path unchanged.
pub inline fn lmHeadProject(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
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
pub inline fn quantParamsFor(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array) QuantParams {
    return self.quantParamsHinted(w, sc, null);
}

/// Resolve per-weight (bits, group_size, mode) with a lazy cache keyed by
/// the scales array pointer. First touch computes params from the scales
/// dtype + shapes (~4 FFI calls); subsequent calls are a single pointer
/// compare. `in_dim` is the weight's input dimension when the caller knows
/// it (activation inner dim) — required to disambiguate (bits, group_size)
/// for affine-override tensors inside a non-affine model.
/// Thread-safety: generation is single-threaded.
pub inline fn quantParamsHinted(self: *const Transformer, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
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
                .mode = @fromBackingInt(@intCast(cache.vals_mode[idx])),
            };
        }
        if (entry == null) {
            const detected = computeQuantParams(.of(&self.config), w, sc, in_dim);
            _ = cache.put(key_raw, detected);
            return detected;
        }
    }
    // Probe window saturated — fall through to direct detect, no cache write.
    return computeQuantParams(.of(&self.config), w, sc, in_dim);
}

pub inline fn qmatmulAddBias(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
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
pub inline fn qmatmulMaybeBias(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
    if (bias.ctx != null) return self.qmatmulAddBias(x, w, sc, bi, bias);
    return self.qmatmul(x, w, sc, bi);
}
