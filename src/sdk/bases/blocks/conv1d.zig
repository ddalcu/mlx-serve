//! A depthwise causal conv over a recurrent layer's conv window: the window (or zeros) prepended, the conv, the
//! window's new tail. The caller reads and writes the state through `CacheOps`.
//! copied_from: sdk/blocks/conv/conv1d_cache.zig (`conv1dWithCache`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const PREFILL_EVAL_MIN_SEQ = @import("../run.zig").PREFILL_EVAL_MIN_SEQ;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const depthwise = @import("../../blocks/conv/depthwise.zig");
const compactConvStateEnabled = @import("../../blocks/conv/depthwise.zig").compactConvStateEnabled;

const Array = ops.Array;
const Graph = ops.Graph;

/// The conv's output and what the cache keeps: the next window and the whole input (a verify's rollback reads it).
pub const Out = struct { y: Array, window: Array, input: Array };

/// `x` `[B, S, C]` after `window` `[B, K-1, C]` (null = a cold layer: zeros), through the depthwise `w` (+ `bias`),
/// SiLU when `silu`. Every array is owned by `g`.
pub fn run(g: *Graph, x: Array, window: ?Array, w: Array, bias: ?Array, batch: c_int, cdim: c_int, kernel: c_int, silu: bool) !Out {
    const prev = window orelse try g.zeros(&.{ batch, kernel - 1, cdim }, ops.dtype(x));
    const input = try g.concat(&.{ prev, x }, 1);
    const len = ops.shape(input)[1];
    var tail = try g.slice(input, &.{ 0, len - (kernel - 1), 0 }, &.{ batch, len, cdim });
    // An owned copy: a slice of a chunk-wide input keeps the whole chunk alive until the layer's next forward.
    if (compactConvStateEnabled() and len - (kernel - 1) >= PREFILL_EVAL_MIN_SEQ) {
        tail = try g.adopt(try materializedOwnedCopy(g.s, tail));
        depthwise.compact_conv_state_copy_count += 1;
    }
    var y = if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(g.s) and len == kernel)
        try g.adopt(try depthwise.depthwiseConvStep(input, w, cdim, kernel, g.s))
    else if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(g.s))
        try g.adopt(try depthwise.depthwiseConvCuda(input, w, cdim, kernel, g.s))
    else
        try g.conv1d(input, w, 1, 0, 1, cdim);
    if (bias) |b| y = try g.add(y, b);
    if (silu) y = try g.mul(y, try g.sigmoid(y));
    return .{ .y = y, .window = tail, .input = input };
}
