//! A causal depthwise convolution over the slot's cached window: the conv state (the last `kernel - 1` input rows)
//! is read and replaced through `CacheOps`, never held by the block; a spec verify also leaves the whole input
//! (`StateSlot.conv_input`). Shared by the gated conv (LFM2) and Mamba2.
//! copied_from: sdk/blocks/conv/conv1d_cache.zig (`conv1dWithCache`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../block.zig");
const kv = @import("../kv.zig");
const PREFILL_EVAL_MIN_SEQ = @import("../run.zig").PREFILL_EVAL_MIN_SEQ;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const depthwise = @import("../../blocks/conv/depthwise.zig");
const compactConvStateEnabled = @import("../../blocks/conv/depthwise.zig").compactConvStateEnabled;

const Array = ops.Array;
const Ctx = block.Ctx;

/// `x` `[1, S, cdim]` convolved with `w` `[cdim, kernel, 1]` behind the cached window (zeros before the first
/// write), plus `bias`, then SiLU when asked. The new window replaces the slot's conv state.
pub fn conv(c: *Ctx, x: Array, w: Array, bias: Array, cdim: c_int, kernel: c_int, silu: bool) !Array {
    const g = c.g;
    const prev = (try kv.state(c, .conv)) orelse try g.zeros(&.{ 1, kernel - 1, cdim }, ops.dtype(x));
    const input = try g.concat(&.{ prev, x }, 1);

    const len = ops.shape(input)[1];
    var tail = try g.sliceAxis(input, 1, len - (kernel - 1), len);
    // A chunk-wide parent would stay alive behind a slice of it until the layer's next forward.
    if (compactConvStateEnabled() and len - (kernel - 1) >= PREFILL_EVAL_MIN_SEQ) {
        tail = try g.adopt(try materializedOwnedCopy(g.s, tail));
        depthwise.compact_conv_state_copy_count += 1;
    }
    try kv.setState(c, .conv, tail);
    if (c.capture_spec) try kv.setState(c, .conv_input, input);

    var out = if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(g.s) and len == kernel)
        try g.adopt(try depthwise.depthwiseConvStep(input, w, cdim, kernel, g.s))
    else if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(g.s))
        try g.adopt(try depthwise.depthwiseConvCuda(input, w, cdim, kernel, g.s))
    else
        try g.conv1d(input, w, 1, 0, 1, cdim);
    if (bias.ctx != null) out = try g.add(out, bias);
    return if (silu) g.mul(out, try g.sigmoid(out)) else out;
}
