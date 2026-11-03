//! engines/mlx/methods/attn_proj: `Transformer` methods (aliased from engines/mlx/transformer.zig).
//! Owns qkAngleFor, qkAngleRowsFor, attnDqFor, attnProj, attnProjBias.

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const decode_attn_quant = @import("../../../sdk/blocks/attention/decode_attn_quant.zig");
const fused_rows = @import("../../../sdk/blocks/linear/fused_rows.zig");
const lastDim = @import("../../../mlx/array.zig").lastDim;
const Transformer = @import("../transformer.zig").Transformer;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const ropeAngleRows = @import("../../../sdk/blocks/rope/qk_norm_rope.zig").ropeAngleRows;
const AttnDqCopy = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").AttnDqCopy;
const attnDqUseNvfp4 = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").attnDqUseNvfp4;
const buildAttnDqCopy = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").buildAttnDqCopy;
const decodeAttnQuantEligible = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").decodeAttnQuantEligible;
const decodeAttnQuantEnabled = @import("../../../sdk/blocks/attention/decode_attn_quant.zig").decodeAttnQuantEnabled;

// ── Laguna attention (per-layer heads, YaRN/default RoPE, softplus gate) ──
// Full-attention layers: 48 Q-heads, YaRN RoPE (theta 5e5, partial 0.5),
// no sliding window. Sliding layers: 72 Q-heads, default RoPE (theta 1e4,
// full rotary), 512-window mask. KV heads uniform (8). Per-head QK RMS-norm
// before RoPE (qwen3 pattern); softplus per-head output gate before o_proj.
// Mirrors gemma4MoeAttnWith's sliding-mask plumbing (head_dim 128 → no
// fused hd-256 kernel). Reference: modeling_laguna.py LagunaAttention.

/// The angle row for `offset` in the given rope family, rebuilt when the
/// offset moves (per decode token: 38-of-40 layers hit the cache).
pub fn qkAngleFor(self: *Transformer, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, mscale: f32) !mlx.mlx_array {
    return self.qkAngleRowsFor(family, rd, base, freqs, offset, 1, mscale, 1.0);
}

/// Multi-row variant for the hd-256 fused path at verify widths: one
/// [rows, rd] table for positions offset..offset+rows-1, same per-forward
/// cache (every layer of one verify round shares the table).
///
/// `mscale` is the YaRN attention factor folded into the rows (1.0 = no
/// scaling). The kernels take cos|sin rows and nothing else, and YaRN's
/// mscale is one uniform scalar over the rotated slice, so folding it here
/// reaches both halves of every pair — the fused path needs no mscale slot
/// of its own. It is baked into the cached rows, which stay keyed by
/// family: a scaled model ropes in family 1, so a scaled row and an
/// unscaled one can never be served to each other.
pub fn qkAngleRowsFor(self: *Transformer, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, rows: c_int, mscale: f32, scale: f32) !mlx.mlx_array {
    const slot = &self.qk_angle_cache[family];
    if (slot.arr.ctx != null and slot.offset == offset and slot.rd == rd and slot.rows == rows and slot.scale == scale) return slot.arr;
    if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
    slot.arr = .{ .ctx = null };
    slot.arr = try ropeAngleRows(self.s, rd, base, freqs, offset, rows, self.allocator, mscale, scale);
    slot.offset = offset;
    slot.rd = rd;
    slot.rows = rows;
    slot.scale = scale;
    return slot.arr;
}

/// Side copy for a dense attention weight, built on first decode use and
/// cached by the weight's ctx pointer (an ineligible weight caches its
/// refusal so the probe runs once). `layer` picks the copy's quantization:
/// int8-g32 affine for the body, real nvfp4-g16 for the tail (see
/// `attnDqUseNvfp4`) — each weight belongs to exactly one layer, so the
/// pointer-keyed cache stays mode-stable.
pub fn attnDqFor(self: *Transformer, w: mlx.mlx_array, layer: u32) !?AttnDqCopy {
    const key = @intFromPtr(w.ctx orelse return null);
    if (self.attn_dq.get(key)) |slot| return slot;
    if (!decodeAttnQuantEligible(mlx.getShape(w), mlx.mlx_array_dtype(w), false)) {
        try self.attn_dq.put(self.allocator, key, null);
        log.debug("[decode-attn-quant] declined a projection: shape/dtype ineligible\n", .{});
        return null;
    }
    const nvfp4 = attnDqUseNvfp4(layer, @intCast(self.config.num_hidden_layers));
    const copy = (try buildAttnDqCopy(self.s, w, nvfp4)) orelse {
        try self.attn_dq.put(self.allocator, key, null);
        return null;
    };
    try self.attn_dq.put(self.allocator, key, copy);
    if (!decode_attn_quant.attn_dq_engaged) {
        decode_attn_quant.attn_dq_engaged = true;
        log.info("[decode-attn-quant] engaged: dense attention projections -> int8 g32 side copies (decode-only, lossy; --no-decode-attn-quant restores dense)\n", .{});
    }
    if (nvfp4 and !decode_attn_quant.attn_dq_nvfp4_engaged) {
        decode_attn_quant.attn_dq_nvfp4_engaged = true;
        const nl: u32 = @intCast(self.config.num_hidden_layers);
        log.info("[decode-attn-quant] nvfp4-g16 tail engaged from layer {d}/{d}\n", .{ decode_attn_quant.attnDqNvfp4From(nl), nl });
    }
    return copy;
}

/// Attention projection with the decode-only requant fast path.
/// `decode_shape` is the caller's `batch == 1 and !is_prefill` — serial
/// decode AND spec-verify forwards route here (a verify that read dense
/// weights would score drafts against a different distribution than the
/// decode steps produced them from, making PLD-on vs PLD-off diverge at
/// temp 0); prefill and batched forwards keep the dense weight.
pub inline fn attnProj(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, decode_shape: bool, layer: u32) !mlx.mlx_array {
    if (decode_shape and sc.ctx == null and decodeAttnQuantEnabled()) {
        // A build failure falls back to the dense weight, never errors the
        // request out.
        if (self.attnDqFor(w, layer) catch null) |dq| {
            return qmatmulBits(x, dq.w, dq.s, dq.b, dq.bits, dq.group_size, dq.mode, self.s);
        }
    }
    return self.qmatmul(x, w, sc, bi);
}

/// `attnProj` plus an ADDITIVE projection bias when one is present.
/// Separate from `qmatmulMaybeBias` so the decode-attn-quant side copy
/// still applies underneath — the bias is added to whatever the projection
/// produced, quantized or not.
pub inline fn attnProjBias(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bias: mlx.mlx_array, decode_shape: bool, layer: u32) !mlx.mlx_array {
    const mm = try self.attnProj(x, w, sc, bi, decode_shape, layer);
    if (bias.ctx == null) return mm;
    defer _ = mlx.mlx_array_free(mm);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, mm, bias, self.s));
    return out;
}
