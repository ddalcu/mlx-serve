//! mlx/array.
//! Owns takeContig, materializedOwnedCopy, concatAxis0, axisView, concatAxis1, sliceLastDim, and 4 more.

const std = @import("std");
const mlx = @import("ffi.zig");
const testing = std.testing;
const mlxSettledActiveBytes = @import("../sdk/testing/parity.zig").mlxSettledActiveBytes;

/// take_axis + materialized copy: quantized kernels (dequantize included)
/// misread the strided view take hands back, so the gather-then-dequantize
/// path must own contiguous rows first. NOTE: not the `+0` trick — an
/// add-materialized take of packed rows still reads back wrong, so this uses
/// the canonical contiguous primitive.
pub fn takeContig(s: mlx.mlx_stream, src: mlx.mlx_array, idx: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    // One owner for `view`: an `errdefer` here plus a `defer` after the take double-freed it
    // when `mlx_contiguous` failed (reachable since #353 turned exit(-1) into an unwind).
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_take_axis(&view, src, idx, axis, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, view, false, s));
    return out;
}

/// Force a REAL copy of `x` into a freshly allocated buffer. `mlx_copy` and
/// `mlx_contiguous` both take shared-buffer fast paths (Copy is a view op;
/// contiguous no-ops when the view's strides read as row-contiguous, which a
/// size-1-leading-dim slice does), so neither breaks the alias to a slice's
/// parent buffer. Adding a same-dtype scalar zero always runs a kernel with a
/// newly allocated output. Buffer donation can't alias either: the caller
/// still holds `x`, so its buffer is never donated.
pub fn materializedOwnedCopy(s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    const scalar_shape = [_]c_int{1};
    try mlx.check(mlx.mlx_zeros(&zero, &scalar_shape, 0, mlx.mlx_array_dtype(x), s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, x, zero, s));
    return out;
}

pub fn concatAxis0(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

/// Index `i` of `arr` along `axis` as an owned view handle (the axis keeps size 1).
pub fn axisView(s: mlx.mlx_stream, arr: mlx.mlx_array, axis: usize, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    for (0..shape.len) |d| stop[d] = shape[d];
    start[axis] = @intCast(i);
    stop[axis] = @intCast(i + 1);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn concatAxis1(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

pub fn concatRows(parts: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A 0-d scalar in `dt` (a bare f32 scalar array promotes bf16 operands).
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

pub fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

/// Last-axis size of an array, as the input-dimension hint for
/// `quantParamsHinted`. Null for null-ctx handles or degenerate shapes.
pub inline fn lastDim(x: mlx.mlx_array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0) return null;
    const d = shape[shape.len - 1];
    return if (d > 0) @as(u32, @intCast(d)) else null;
}

pub fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // Swap the last two axes for any rank: 2D weights [out, in] → [in, out];
    // stacked MoE expert tensors [experts, out, in] → [experts, in, out].
    const ndim = mlx.getShape(w).len;
    var perm: [8]c_int = undefined; // mlx arrays never exceed 8 dims here
    for (0..ndim) |i| perm[i] = @intCast(i);
    perm[ndim - 2] = @intCast(ndim - 1);
    perm[ndim - 1] = @intCast(ndim - 2);
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

pub fn splitPackedGateUp(arr: mlx.mlx_array, s: mlx.mlx_stream) !struct { gate: mlx.mlx_array, up: mlx.mlx_array } {
    const shape = mlx.getShape(arr);
    if (shape.len != 3) return error.BadPackedGateUpShape;
    const two_m = shape[1];
    const m = @divExact(two_m, 2);
    const strides = [_]c_int{ 1, 1, 1 };

    var gate_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_view);
    const g_start = [_]c_int{ 0, 0, 0 };
    const g_stop = [_]c_int{ shape[0], m, shape[2] };
    try mlx.check(mlx.mlx_slice(&gate_view, arr, &g_start, 3, &g_stop, 3, &strides, 3, s));

    var up_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_view);
    const u_start = [_]c_int{ 0, m, 0 };
    const u_stop = [_]c_int{ shape[0], two_m, shape[2] };
    try mlx.check(mlx.mlx_slice(&up_view, arr, &u_start, 3, &u_stop, 3, &strides, 3, s));

    // gather_qmm reads the expert weight buffers as ROW-CONTIGUOUS — a lazy
    // slice view has parent strides and silently produces zeros. Materialize
    // both halves (one-time cost at load).
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_contiguous(&gate, gate_view, false, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_contiguous(&up, up_view, false, s));

    return .{ .gate = gate, .up = up };
}

// ── tests ──

/// `arr[:, :, start..stop]` as a lazy view.
pub fn sliceLastDim(arr: mlx.mlx_array, start: c_int, stop: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    if (sh.len != 3) return error.BadShape;
    const st = [_]c_int{ 0, 0, start };
    const en = [_]c_int{ sh[0], sh[1], stop };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, arr, &st, 3, &en, 3, &strides, 3, s));
    return out;
}

test "takeContig releases `view` exactly once on every faulted op (CPU stream, no GPU)" {
    // Driven with the fault injector at `mlx.check`: `arm(k)` fails the k-th checked call, so
    // the sweep walks every error path. On the pre-fix shape k = 2 double-frees and aborts.
    const s = mlx.mlx_default_cpu_stream_new();
    const rows = 8;
    const cols = 4;
    var host: [rows * cols]f32 = undefined;
    for (&host, 0..) |*v, n| v.* = @floatFromInt(n);
    const shape = [_]c_int{ rows, cols };
    const srcarr = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(srcarr);
    var idx_host = [_]u32{ 5, 0, 3, 3 };
    const ish = [_]c_int{idx_host.len};
    const idx = mlx.mlx_array_new_data(&idx_host, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(idx);

    const c0 = mlx.op_count.load(.monotonic);
    const warm = try takeContig(s, srcarr, idx, 0);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    try mlx.check(mlx.mlx_array_eval(warm));
    try testing.expectEqual(@as(c_int, idx_host.len), mlx.getShape(warm)[0]);
    _ = mlx.mlx_array_free(warm);
    try testing.expectEqual(@as(u64, 2), n_ops);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        mlx.fault.arm(k);
        const r = takeContig(s, srcarr, idx, 0);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2), fired);
}

test "mlx check fault injection: the ownership shape releases its array on every faulted op" {
    // The machinery behind the fixture-gated sweep below, exercised on a shape
    // that holds nothing but the ownership pattern of `qwen4MtpForward`'s `h`:
    // an array born from an op, guarded by ONE function-scope errdefer,
    // rebound through a BLOCK-scoped slice that frees its predecessor itself,
    // then handed to a further fallible op. Every array here is eval'd, so a
    // leaked handle holds real Metal bytes and the active-memory oracle can
    // see it. The `leaky` arm is the NEGATIVE CONTROL: the same shape minus
    // the one errdefer must trip the oracle, or a green sweep means nothing.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const H = struct {
        fn guarded(st: mlx.mlx_stream, seed: mlx.mlx_array) !mlx.mlx_array {
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&h, seed, seed, st));
            errdefer _ = mlx.mlx_array_free(h);
            try mlx.check(mlx.mlx_array_eval(h));
            {
                var h_last = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(h_last);
                try mlx.check(mlx.mlx_add(&h_last, h, seed, st));
                try mlx.check(mlx.mlx_array_eval(h_last));
                _ = mlx.mlx_array_free(h);
                h = h_last;
            }
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_multiply(&out, h, seed, st));
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(h);
            return out;
        }
        /// Same shape, ONE errdefer short. `h` lives in the CALLER's frame, so
        /// the test can free whatever this function drops after measuring it —
        /// an mlx op OVERWRITES its destination handle, so a copy taken before
        /// the op is a stale pointer, not a rescue line.
        fn leaky(st: mlx.mlx_stream, seed: mlx.mlx_array, h: *mlx.mlx_array) !mlx.mlx_array {
            try mlx.check(mlx.mlx_multiply(h, seed, seed, st));
            try mlx.check(mlx.mlx_array_eval(h.*));
            {
                var h_last = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(h_last);
                try mlx.check(mlx.mlx_add(&h_last, h.*, seed, st));
                try mlx.check(mlx.mlx_array_eval(h_last));
                _ = mlx.mlx_array_free(h.*);
                h.* = h_last;
            }
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_multiply(&out, h.*, seed, st));
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(h.*);
            h.* = .{ .ctx = null };
            return out;
        }
    };

    const n_elem: usize = 1 << 20; // 4 MB per array: far above measurement noise
    const host = try allocator.alloc(f32, n_elem);
    defer allocator.free(host);
    @memset(host, 1.5);
    const shape = [_]c_int{@intCast(n_elem)};
    const seed = mlx.mlx_array_new_data(host.ptr, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(seed);
    try mlx.check(mlx.mlx_array_eval(seed));

    // One un-faulted run: warms the allocator and counts the checked ops the
    // shape issues, which is the sweep's upper bound. Its output is freed here.
    const c0 = mlx.op_count.load(.monotonic);
    const warm = try H.guarded(s, seed);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    _ = mlx.mlx_array_free(warm);
    try testing.expect(n_ops >= 6);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        const base = mlxSettledActiveBytes(s);
        mlx.fault.arm(k);
        const r = H.guarded(s, seed);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
        }
        // A DOUBLE free would abort this binary here rather than fail an
        // assertion — that abort is the other half of the proof.
        const after = mlxSettledActiveBytes(s);
        if (after != base) std.debug.print("[fault] guarded k={d}: base {d} after {d}\n", .{ k, base, after });
        try testing.expectEqual(base, after);
    }
    try testing.expectEqual(n_ops, fired);

    // Negative control: the same sweep without the errdefer must be CAUGHT.
    var caught: usize = 0;
    k = 1;
    while (k <= n_ops) : (k += 1) {
        var probe = mlx.mlx_array_new();
        const base = mlxSettledActiveBytes(s);
        mlx.fault.arm(k);
        const r = H.leaky(s, seed, &probe);
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |_| {}
        if (mlxSettledActiveBytes(s) > base + (n_elem * @sizeOf(f32)) / 2) caught += 1;
        if (probe.ctx != null) _ = mlx.mlx_array_free(probe);
        const after_ctl = mlxSettledActiveBytes(s);
        if (after_ctl != base) std.debug.print("[fault] leaky k={d}: base {d} after {d}\n", .{ k, base, after_ctl });
        try testing.expectEqual(base, after_ctl);
    }
    std.debug.print("[fault] ownership shape: {d} checked ops, {d} faulted cleanly, leak oracle caught {d}\n", .{ n_ops, fired, caught });
    try testing.expect(caught >= 1);
}
