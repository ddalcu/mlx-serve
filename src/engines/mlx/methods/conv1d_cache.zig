//! engines/mlx/methods/conv1d_cache: `Transformer` methods (aliased from engines/mlx/transformer.zig).
//! Owns conv1dWithCache, conv1dTailRetentionDelta, conv1dBitCopy, conv1dTwoChunk, conv1dExpectBitEqual.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const PREFILL_EVAL_MIN_SEQ = @import("../../../sdk/bases/run.zig").PREFILL_EVAL_MIN_SEQ;
const compactConvStateEnabled = @import("../../../sdk/blocks/conv/depthwise.zig").compactConvStateEnabled;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../transformer.zig").Transformer;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const depthwiseConvCuda = @import("../../../sdk/blocks/conv/depthwise.zig").depthwiseConvCuda;
const depthwiseConvStep = @import("../../../sdk/blocks/conv/depthwise.zig").depthwiseConvStep;
const depthwise_vars = @import("../../../sdk/blocks/conv/depthwise.zig");

// ── Conv1d with cache (shared by GatedDeltaNet, LFM2, Mamba2) ──

/// Prepends cached conv state, applies depthwise conv1d, updates cache.
/// If apply_silu is true, applies SiLU activation after conv.
/// conv_b is an optional bias added after conv1d.
pub fn conv1dWithCache(
    self: *Transformer,
    x: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    conv_b: ?mlx.mlx_array,
    ssm: *SSMCacheEntry,
    batch: c_int,
    cdim: c_int,
    kernel: c_int,
    apply_silu: bool,
) !mlx.mlx_array {
    // Prepend conv_state or zeros
    var conv_input: mlx.mlx_array = undefined;
    defer _ = mlx.mlx_array_free(conv_input);
    if (ssm.initialized) {
        const arr = [_]mlx.mlx_array{ ssm.conv_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    } else {
        const zero_shape = [_]c_int{ batch, kernel - 1, cdim };
        var zero_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero_state);
        try mlx.check(mlx.mlx_zeros(&zero_state, &zero_shape, 3, mlx.mlx_array_dtype(x), self.s));
        const arr = [_]mlx.mlx_array{ zero_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    }

    // Update conv_state: keep last (kernel-1) positions
    {
        const ci_shape = mlx.getShape(conv_input);
        const ci_len = ci_shape[1];
        const keep_start = ci_len - (kernel - 1);
        const start = [_]c_int{ 0, keep_start, 0 };
        const stop = [_]c_int{ batch, ci_len, cdim };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &start, 3, &stop, 3, &strides, 3, self.s));
        errdefer _ = mlx.mlx_array_free(new_conv_state);
        // The tail must be an OWNED copy: a slice of the chunk-wide `conv_input` keeps
        // the whole chunk alive in this layer until its next forward. Only a chunk-wide
        // parent is worth the copy; at decode width the parent is already tiny.
        const t_rows = ci_len - (kernel - 1);
        if (compactConvStateEnabled() and t_rows >= PREFILL_EVAL_MIN_SEQ) {
            const owned = try materializedOwnedCopy(self.s, new_conv_state);
            _ = mlx.mlx_array_free(new_conv_state);
            new_conv_state = owned;
            depthwise_vars.compact_conv_state_copy_count += 1;
        }
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = new_conv_state;
        ssm.initialized = true;
    }

    // PLD capture: stash the full conv input ([B, (kernel-1)+T, conv_dim])
    // so partial-accept rollback can slice the accepted-position conv_state
    // without a re-forward. Refcount-shared; freed at the end of the round.
    if (self.spec_capture_ssm) {
        if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
        ssm.spec_conv_input = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&ssm.spec_conv_input, conv_input);
    }

    // Depthwise conv1d (groups = cdim)
    var conv_out = if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(self.s) and mlx.getShape(conv_input)[1] == kernel)
        try depthwiseConvStep(conv_input, conv_w, cdim, kernel, self.s)
    else if (mlx.cudaAvailable() and mlx.streamOnGpuDevice(self.s))
        try depthwiseConvCuda(conv_input, conv_w, cdim, kernel, self.s)
    else blk: {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&o, conv_input, conv_w, 1, 0, 1, cdim, self.s));
        break :blk o;
    };

    // Optional bias
    if (conv_b) |cb| {
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, conv_out, cb, self.s));
        _ = mlx.mlx_array_free(conv_out);
        conv_out = biased;
    }

    // Optional SiLU activation
    if (apply_silu) {
        const activated = try self.silu(conv_out);
        _ = mlx.mlx_array_free(conv_out);
        return activated;
    }
    return conv_out;
}

// ── tests ──

fn conv1dTailRetentionDelta(batch: c_int, seq: c_int, cdim: c_int, kernel: c_int) !usize {
    const t = std.testing;
    const s = mlx.gpuStream();
    const saved = depthwise_vars.compact_conv_state_override;
    depthwise_vars.compact_conv_state_override = true;
    defer depthwise_vars.compact_conv_state_override = saved;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = s;
    xfm.spec_capture_ssm = false;

    var prng = std.Random.DefaultPrng.init(0xC0A1_7A11);
    const x = try attn256RandBf16(prng.random(), &[_]c_int{ batch, seq, cdim }, s);
    var x_live = true;
    defer if (x_live) {
        _ = mlx.mlx_array_free(x);
    };
    var w = mlx.mlx_array_new();
    var w_live = true;
    defer if (w_live) {
        _ = mlx.mlx_array_free(w);
    };
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));

    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var baseline: usize = 0;
    _ = mlx.mlx_get_active_memory(&baseline);

    const out = try xfm.conv1dWithCache(x, w, null, &entry, batch, cdim, kernel, false);
    var out_live = true;
    defer if (out_live) {
        _ = mlx.mlx_array_free(out);
    };
    try mlx.check(mlx.mlx_array_eval(out));
    try mlx.check(mlx.mlx_array_eval(entry.conv_state));
    out_live = false;
    x_live = false;
    w_live = false;
    _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_free(x);
    _ = mlx.mlx_array_free(w);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    const delta = if (after > baseline) after - baseline else 0;
    try t.expect(entry.initialized);
    const keep = kernel - 1;
    const sh = mlx.getShape(entry.conv_state);
    try t.expectEqual(@as(c_int, batch), sh[0]);
    try t.expectEqual(keep, sh[1]);
    try t.expectEqual(cdim, sh[2]);
    return delta;
}

// Bar: after the conv, freeing the chunk input leaves only the tail resident.
test "conv1dWithCache tail drops the prefill chunk parent" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const batch: c_int = 1;
    const seq: c_int = 4096;
    const cdim: c_int = 256;
    const kernel: c_int = 4;
    const delta = try conv1dTailRetentionDelta(batch, seq, cdim, kernel);
    const chunk: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(seq)) * @as(usize, @intCast(cdim)) * 2;
    if (delta >= 64 * 1024) std.debug.print("retention delta={d} chunk={d}\n", .{ delta, chunk });
    try t.expect(delta < 64 * 1024);
}

// Bar: same as above with a strided batch-2 parent.
test "conv1dWithCache tail drops a batch-2 strided prefill parent" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const batch: c_int = 2;
    const seq: c_int = 4096;
    const cdim: c_int = 256;
    const kernel: c_int = 4;
    const delta = try conv1dTailRetentionDelta(batch, seq, cdim, kernel);
    const chunk: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(seq)) * @as(usize, @intCast(cdim)) * 2;
    if (delta >= 64 * 1024) std.debug.print("retention delta={d} chunk={d}\n", .{ delta, chunk });
    try t.expect(delta < 64 * 1024);
}

fn conv1dBitCopy(batch: c_int, seq: c_int, cdim: c_int, kernel: c_int, compact: bool, s: mlx.mlx_stream) !struct { out: mlx.mlx_array, conv_state: mlx.mlx_array } {
    const saved = depthwise_vars.compact_conv_state_override;
    depthwise_vars.compact_conv_state_override = compact;
    defer depthwise_vars.compact_conv_state_override = saved;

    var xfm: Transformer = undefined;

    xfm.rht = null;
    xfm.s = s;
    xfm.spec_capture_ssm = false;
    var prng = std.Random.DefaultPrng.init(0xB17E_C0DE);
    const x = try attn256RandBf16(prng.random(), &[_]c_int{ batch, seq, cdim }, s);
    defer _ = mlx.mlx_array_free(x);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    errdefer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const out = try xfm.conv1dWithCache(x, w, null, &entry, batch, cdim, kernel, true);
    return .{ .out = out, .conv_state = entry.conv_state };
}

fn conv1dTwoChunk(batch: c_int, t1: c_int, t2: c_int, cdim: c_int, kernel: c_int, compact: bool, s: mlx.mlx_stream) !struct { out: mlx.mlx_array, conv_state: mlx.mlx_array } {
    const saved = depthwise_vars.compact_conv_state_override;
    depthwise_vars.compact_conv_state_override = compact;
    defer depthwise_vars.compact_conv_state_override = saved;

    var xfm: Transformer = undefined;

    xfm.rht = null;
    xfm.s = s;
    xfm.spec_capture_ssm = false;
    var prng = std.Random.DefaultPrng.init(0xA11C_EDED);
    const x1 = try attn256RandBf16(prng.random(), &[_]c_int{ batch, t1, cdim }, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try attn256RandBf16(prng.random(), &[_]c_int{ batch, t2, cdim }, s);
    defer _ = mlx.mlx_array_free(x2);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    errdefer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const out1 = try xfm.conv1dWithCache(x1, w, null, &entry, batch, cdim, kernel, true);
    _ = mlx.mlx_array_free(out1);
    const out2 = try xfm.conv1dWithCache(x2, w, null, &entry, batch, cdim, kernel, true);
    return .{ .out = out2, .conv_state = entry.conv_state };
}

fn conv1dExpectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try std.testing.expect(ok);
}

// Bar: output and stored state byte-equal with the copy on and off.
test "conv1dWithCache compact conv_state is bit-identical on and off" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const kernel: c_int = 4;
    const cdim: c_int = 32;
    const cases = [_][2]c_int{
        .{ 1, 1 },
        .{ 1, 5 },
        .{ 1, 4096 },
        .{ 2, 1 },
        .{ 2, 5 },
        .{ 2, 4096 },
    };
    for (cases) |c| {
        const batch = c[0];
        const seq = c[1];
        const off = try conv1dBitCopy(batch, seq, cdim, kernel, false, s);
        defer {
            _ = mlx.mlx_array_free(off.out);
            _ = mlx.mlx_array_free(off.conv_state);
        }
        const on = try conv1dBitCopy(batch, seq, cdim, kernel, true, s);
        defer {
            _ = mlx.mlx_array_free(on.out);
            _ = mlx.mlx_array_free(on.conv_state);
        }
        try conv1dExpectBitEqual(off.out, on.out, s);
        try conv1dExpectBitEqual(off.conv_state, on.conv_state, s);
    }
}

// Bar: below the chunk bar the tail stays a view (no copy kernel per decode step).
test "conv1dWithCache decode width does not copy the tail" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    const saved = depthwise_vars.compact_conv_state_override;
    defer depthwise_vars.compact_conv_state_override = saved;
    depthwise_vars.compact_conv_state_override = true;
    depthwise_vars.compact_conv_state_copy_count = 0;
    const on = try conv1dBitCopy(1, 1, 32, 4, true, s);
    defer {
        _ = mlx.mlx_array_free(on.out);
        _ = mlx.mlx_array_free(on.conv_state);
    }
    try t.expectEqual(@as(u64, 0), depthwise_vars.compact_conv_state_copy_count);
    const off = try conv1dBitCopy(1, 1, 32, 4, false, s);
    defer {
        _ = mlx.mlx_array_free(off.out);
        _ = mlx.mlx_array_free(off.conv_state);
    }
    try conv1dExpectBitEqual(on.out, off.out, s);
    try conv1dExpectBitEqual(on.conv_state, off.conv_state, s);
}

// Bar: the compacted state prepended to the next chunk gives byte-equal results.
test "conv1dWithCache compact conv_state is bit-identical across two chunks" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const off = try conv1dTwoChunk(1, 64, 64, 32, 4, false, s);
    defer {
        _ = mlx.mlx_array_free(off.out);
        _ = mlx.mlx_array_free(off.conv_state);
    }
    const on = try conv1dTwoChunk(1, 64, 64, 32, 4, true, s);
    defer {
        _ = mlx.mlx_array_free(on.out);
        _ = mlx.mlx_array_free(on.conv_state);
    }
    try conv1dExpectBitEqual(off.out, on.out, s);
    try conv1dExpectBitEqual(off.conv_state, on.conv_state, s);
}
