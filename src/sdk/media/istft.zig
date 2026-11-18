//! The ISTFT head of the vocoder-style decoders (Kokoro, Ming-3.0-tts): a complex spectrum to time-domain frames on the
//! GPU, then a host overlap-add divided by the window envelope and trimmed at both ends.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");

/// Samples from `spec` [1, frames, n_fft / 2 + 1] (complex, frequency on the last axis): irfft to frames, then
/// `overlapAdd`. The caller builds `spec` in its reference's op order and frees the result.
pub fn istft(a: std.mem.Allocator, spec: mlx.mlx_array, n_fft: usize, hop: usize, window: []const f32, pad: usize, s: mlx.mlx_stream) ![]f32 {
    var fr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fr);
    try mlx.check(mlx.mlx_fft_irfft(&fr, spec, @intCast(n_fft), -1, mlx.MLX_FFT_NORM_BACKWARD, s));
    try mlx.check(mlx.mlx_array_eval(fr));
    const n = mlx.mlx_array_size(fr) / n_fft;
    const data = mlx.mlx_array_data_float32(fr) orelse return error.IstftReadFailed;
    return overlapAdd(a, data[0 .. n * n_fft], n, n_fft, hop, window, pad);
}

/// Overlap-add of `frames` rows of `frames_data` [frames, n_fft], each times `window`, divided by the summed squared
/// window and trimmed by `pad` at both ends: `n_fft / 2` for `torch.istft(center=True)`, `(n_fft - hop) / 2` for
/// "same" padding. Frames add in order, so the sums match a reference that loops over frames.
pub fn overlapAdd(a: std.mem.Allocator, frames_data: []const f32, frames: usize, n_fft: usize, hop: usize, window: []const f32, pad: usize) ![]f32 {
    if (frames == 0) return a.alloc(f32, 0);
    const padded_len = (frames - 1) * hop + n_fft;
    if (2 * pad > padded_len) return error.IstftPadTooLong;
    const acc = try a.alloc(f32, padded_len);
    defer a.free(acc);
    const wsum = try a.alloc(f32, padded_len);
    defer a.free(wsum);
    @memset(acc, 0);
    @memset(wsum, 0);
    for (0..frames) |f| {
        const off = f * hop;
        for (0..n_fft) |k| {
            acc[off + k] += frames_data[f * n_fft + k] * window[k];
            wsum[off + k] += window[k] * window[k];
        }
    }
    const out = try a.alloc(f32, padded_len - 2 * pad);
    for (out, 0..) |*o, i| {
        const w = wsum[pad + i];
        // torch guards the same way; below this the frame coverage is degenerate.
        o.* = if (w > 1e-11) acc[pad + i] / w else 0;
    }
    return out;
}

// ── tests ──

const testing = std.testing;

fn hann(w: []f32) void {
    const n: f32 = @floatFromInt(w.len);
    for (w, 0..) |*v, i| v.* = 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f32, @floatFromInt(i)) / n);
}

test "istft: windowed frames overlap-add back to the signal, for both trims" {
    const n_fft = 20;
    const hop = 5;
    const n = 37;
    var w: [n_fft]f32 = undefined;
    hann(&w);
    var x: [(n - 1) * hop + n_fft]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.3) * 0.7 + 0.2;
    var frames: [n * n_fft]f32 = undefined;
    for (0..n) |f| for (0..n_fft) |k| {
        frames[f * n_fft + k] = x[f * hop + k] * w[k];
    };
    for ([_]usize{ n_fft / 2, (n_fft - hop) / 2 }) |pad| {
        const y = try overlapAdd(testing.allocator, &frames, n, n_fft, hop, &w, pad);
        defer testing.allocator.free(y);
        try testing.expectEqual(x.len - 2 * pad, y.len);
        for (y, x[pad..][0..y.len]) |got, want| try testing.expectApproxEqAbs(want, got, 1e-5);
    }
}

test "istft: the irfft of a frame's spectrum overlap-adds like the frame itself" {
    const n_fft = 16;
    const hop = 4;
    const n = 9;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var w: [n_fft]f32 = undefined;
    hann(&w);
    var frames: [n * n_fft]f32 = undefined;
    for (&frames, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.21);
    const shape = [_]c_int{ 1, n, n_fft };
    const t = mlx.mlx_array_new_data(&frames, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(t);
    var spec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(spec);
    try mlx.check(mlx.mlx_fft_rfft(&spec, t, n_fft, -1, mlx.MLX_FFT_NORM_BACKWARD, s));
    const pad = (n_fft - hop) / 2;
    const got = try istft(testing.allocator, spec, n_fft, hop, &w, pad, s);
    defer testing.allocator.free(got);
    const want = try overlapAdd(testing.allocator, &frames, n, n_fft, hop, &w, pad);
    defer testing.allocator.free(want);
    try testing.expectEqual(want.len, got.len);
    for (want, got) |e, o| try testing.expectApproxEqAbs(e, o, 1e-5);
}
