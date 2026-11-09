//! What a wide prompt call's MoE does not read, committed behind its routes before the routing barrier waits on them.
//!
//! Without it the call's barrier evaluates its routes (the layer's attention, its FFN mix and the router) and waits; the
//! next wide layer's predicted routes start behind the barrier once its ids are on the host,
//! and the layer's shared expert and attention state are evaluated only by the layer's last eval, after the call's
//! groups: the GPU idles through the barrier's host work and the call's first read wait, then runs the shared expert's
//! GEMMs (about 25 ms of a 16K-row layer) at the layer's end.
//!
//! So the model's `moeExpand` commits three evaluations before the call, each its
//! own commit: the routes, cast to the int32 ids the streamer reads (its own cast, made here instead); the next layer's
//! predicted routes, when there are any (the next layer's staging then finds them under way); and the shared expert with the
//! layer's attention state. The barrier waits on the first only. The GPU runs the other two during the barrier's host
//! work and the call's first read wait, and the layer's last eval is left the tail kernel. The same kernels run over
//! the same inputs; only independent ones run earlier, so the same words. A prompt call's MoE stage then also holds the
//! shared expert's evaluation (every array of it is made when the host encodes it ahead of the call: 1.03 GB at 16K
//! rows, `bytes`), which the prompt bill counts (the model's `prefillBytes`; only for calls wider than a span, so
//! never the decode phase's bill).
//!
//! Measured (a1 b1 b2 a2, the cold 16K card then a new-text slice):
//! cold 1414.0 / 1418.6 / 1425.1 / 1400.3 tok/s (+1.04%, pairs +0.33% / +1.77%), new text 1490.4 / 1498.1 / 1502.6 /
//! 1486.2 (+0.81%, pairs +0.52% / +1.10%), with 150 prompt rows against 153 (the bill's 1.03 GB) and 177 decode rows in
//! every arm; the early shared expert compared word for word with the one built after the call: 3,205,017,600 words, 0
//! differing; the texts the same bytes.

const std = @import("std");

/// The bytes a wide call's early shared expert holds through its MoE stage, for the prompt bill (the model's
/// `prefillBytes`): every array of its evaluation, allocated when the host encodes it ahead of the call (MLX makes each
/// output then; each is freed only when its consumers have run): the gate/up linear's prepared rows (f16 [rows, d]), its
/// decoded weight (f16 [d, 2n]), its product (f16 [rows, 2n]) and finish (the activation dtype, [rows, 2n]), the activation
/// ([rows, n], and its one f32 weight a row), the down linear's prepared rows (f16 [rows, n]), weight (f16 [n, d]),
/// product (f16 [rows, d]) and finish, the shared rows the tail reads ([rows, d]). `d` the hidden width, `n` the shared
/// expert's (2-byte activations). 1.03 GB at 16K rows of DeepSeek-V4.1 (d 5120, n 2304).
pub fn bytes(rows: u64, d: u64, n: u64) u64 {
    return rows * (2 * d + 4 * n + 4 * n + 2 * n + 4 + 2 * n + 2 * d + 2 * d) + 2 * d * 2 * n + 2 * n * d;
}

test "wide_early bytes: the shared expert's evaluation at 16K rows of DeepSeek-V4.1, about 1.03 GB" {
    const b = bytes(16384, 5120, 2304);
    try std.testing.expectEqual(@as(u64, 16384 * (6 * 5120 + 12 * 2304 + 4) + 6 * 5120 * 2304), b);
    try std.testing.expect(b > 1_020_000_000 and b < 1_035_000_000);
    try std.testing.expectEqual(@as(u64, 6 * 5120 * 2304), bytes(0, 5120, 2304));
}
