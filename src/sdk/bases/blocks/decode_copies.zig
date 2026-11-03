//! The decode-only requantized side copies of attention projections (`--decode-attn-quant`, on by default):
//! `decodeCopy` builds one at load (the engine's `attnDqFor` arithmetic), and `DecodeCopies(B)` wraps an attention
//! block whose q/k/v/o read them at a solo one-row step, as the legacy `attnProj` does; every other width reads the
//! checkpoint's weights.

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const Linear = @import("linear.zig").Linear;
const dq = @import("../../blocks/attention/decode_attn_quant.zig");

const Array = mlx.mlx_array;

/// The decode-only requantized side copy of an attention projection (`--decode-attn-quant`), as the engine's
/// `attnDqFor` builds it from a dense projection: int8 g32, nvfp4 in the tail layers; empty when the projection takes
/// none. Owned by the caller.
pub fn decodeCopy(l: Linear, layer: u32, layers: u32, s: mlx.mlx_stream) !Linear {
    if (!dq.decodeAttnQuantEnabled() or l.s.ctx != null) return .{};
    if (!dq.decodeAttnQuantEligible(ops.shape(l.w), mlx.mlx_array_dtype(l.w), false)) return .{};
    const c = (try dq.buildAttnDqCopy(s, l.w, dq.attnDqUseNvfp4(layer, layers))) orelse return .{};
    return .{ .w = c.w, .s = c.s, .b = c.b, .bits = c.bits, .group_size = c.group_size, .mode = c.mode };
}


pub fn DecodeCopies(comptime B: type) type {
    block.check(B, &.{.mixer});
    return struct {
        pub const kind = B.kind;
        pub const supports_batched_rows = B.supports_batched_rows;
        pub const Weights = struct {
            inner: B.Weights,
            /// q, k, v, o side copies; a null `w` keeps that projection's weight.
            dq: [4]Linear = @splat(.{}),
        };

        pub fn forward(w: *const Weights, c: *block.Ctx, x: ops.Array) !ops.Array {
            if (c.mode == .batched_rows or c.slots != 1 or c.rows != 1 or !dq.decodeAttnQuantEnabled()) return B.forward(&w.inner, c, x);
            var step = w.inner;
            inline for (.{ "q", "k", "v", "o" }, 0..) |name, i| {
                const lin = &@field(step, name);
                if (w.dq[i].w.ctx != null) {
                    const bias = lin.bias;
                    lin.* = w.dq[i];
                    lin.bias = bias;
                }
            }
            return B.forward(&step, c, x);
        }
    };
}
