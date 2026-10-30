//! A sliding layer's reads: the tail of the cache a block of queries can reach (`View`) and the additive band masks
//! over it. copied_from: engines/mlx/kv/sliding.zig (`slidingViewFor`, without its engagement line),
//! sdk/blocks/attention/masks.zig (`createCausalMask`, `createSlidingWindowMask`, `createSlidingWindowDecodeMask`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = @import("../block.zig").Ctx;

pub const View = struct {
    /// Tail rows the cache hands back (`window + q_len - 1`); 0 = the whole cache.
    span: u32,
    /// Keys in that view.
    kv_len: c_int,
    /// The fused hd-256 kernel masks the band from geometry: no mask array is built.
    band_in_kernel: bool,
};

/// The view of a layer of `window` (0 = not sliding) after `q_len` new rows land on `total_kv`.
pub fn view(window: u32, head_dim: c_int, total_kv: c_int, q_len: c_int) View {
    const band = attn256.fused256Enabled() and head_dim == 256 and q_len >= attn256.FUSED256_MIN_Q_LEN;
    const width: usize = if (band) 1 else SLIDING_TRIM_UNBOUNDED;
    const span = slidingTailSpan(window, @intCast(q_len), width);
    const kv_len: c_int = if (span > 0) @min(total_kv, @as(c_int, @intCast(span))) else total_kv;
    return .{ .span = span, .kv_len = kv_len, .band_in_kernel = band };
}

fn negInf(g: *Graph) !Array {
    return g.scalar(-std.math.inf(f32), .bfloat16);
}

/// `[1, 1, q_len, kv_len]` bf16: 0 where key j is at or before query i (the queries are the last `q_len`), -inf after.
pub fn causalMask(parent: *Graph, q_len: c_int, kv_len: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const ones = try g.full(&.{ q_len, kv_len }, try g.scalar(1, .bfloat16), .bfloat16);
    const upper = try g.astype(try g.call(mlx.mlx_triu, .{ ones, kv_len - q_len + 1 }), .bool_);
    const mask = try g.where(upper, try negInf(g), try g.scalar(0, .bfloat16));
    return parent.keep(try g.reshape(mask, &.{ 1, 1, q_len, kv_len }));
}

/// The causal mask plus the band: query i sees keys within `window` behind it.
pub fn bandMask(parent: *Graph, q_len: c_int, kv_len: c_int, window: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const causal = try causalMask(g, q_len, kv_len);
    const off = kv_len - q_len;
    const rows = try g.reshape(try g.arange(@floatFromInt(off), @floatFromInt(off + q_len), 1, .int32), &.{ q_len, 1 });
    const cols = try g.reshape(try g.arange(0, @floatFromInt(kv_len), 1, .int32), &.{ 1, kv_len });
    const too_far = try g.call(mlx.mlx_greater_equal, .{ try g.sub(rows, cols), try g.adopt(mlx.mlx_array_new_int(window)) });
    const band = try g.where(too_far, try negInf(g), try g.scalar(0, .bfloat16));
    return parent.keep(try g.add(causal, try g.reshape(band, &.{ 1, 1, q_len, kv_len })));
}

/// `bandMask`, built once per forward.
pub fn bandOnce(c: *const Ctx, q_len: c_int, kv_len: c_int, window: c_int) !Array {
    const key: @import("../block.zig").Memo.Key = .{ .what = .band, .n = .{ q_len, kv_len, window, 0, 0, 0, 0 } };
    if (c.recall(key)) |m| return m;
    return c.remember(key, try bandMask(c.g, q_len, kv_len, window));
}

/// `decodeMask`, built once per forward.
pub fn decodeOnce(c: *const Ctx, kv_len: c_int, window: c_int) !Array {
    const key: @import("../block.zig").Memo.Key = .{ .what = .decode_mask, .n = .{ kv_len, window, 0, 0, 0, 0, 0 } };
    if (c.recall(key)) |m| return m;
    return c.remember(key, try decodeMask(c.g, kv_len, window));
}

/// One decode row over `kv_len` keys: the oldest `kv_len - window` masked.
pub fn decodeMask(parent: *Graph, kv_len: c_int, window: c_int) !Array {
    var t = parent.scope();
    defer t.deinit();
    const g = &t;
    const pos = try g.arange(0, @floatFromInt(kv_len), 1, .int32);
    const too_old = try g.less(pos, try g.adopt(mlx.mlx_array_new_int(kv_len - window)));
    const mask = try g.where(too_old, try negInf(g), try g.scalar(0, .bfloat16));
    return parent.keep(try g.reshape(mask, &.{ 1, 1, 1, kv_len }));
}

/// "No width bound" for `slidingTailSpan`. A 32-wide cap used to keep prefill
/// chunks out while the mask pairing was unproven; with `band_in_kernel` now
/// gating the only consumer that reads ABSOLUTE positions and every other
/// consumer deriving its query offset as `kv_len - q_len`, width needs no
/// bound — and wide chunks are the only multi-token forward a serial arch
/// (laguna, inkling) ever does.
pub const SLIDING_TRIM_UNBOUNDED: usize = std.math.maxInt(usize);

/// How many TAIL KV entries a forward of `q_len` queries needs from a SLIDING
/// layer. The last query reaches back `window`; the first of the block reaches
/// back `window + q_len - 1`, so the block's union is that many entries.
/// Returns 0 — meaning "hand back the whole cache, no trim" — when the layer
/// does not slide or the block is wider than `max_width`.
///
/// This is what makes a spec verify cost what a decode step costs on a sliding
/// layer: at width 1 it has always returned `window`, but a block-wide verify
/// used to fall through untrimmed and read the ENTIRE cache on every sliding
/// layer, so a 3:1 local:global model paid full attention on 75% of its layers
/// once past the window (measured: muse-glimmer verify 51 -> 140 ms from 0.6k
/// to 17.6k ctx, turning DFlash from 2.28x into 0.74x).
pub fn slidingTailSpan(window: u32, q_len: usize, max_width: usize) u32 {
    if (window == 0 or q_len == 0 or q_len > max_width) return 0;
    return window + @as(u32, @intCast(q_len)) - 1;
}
