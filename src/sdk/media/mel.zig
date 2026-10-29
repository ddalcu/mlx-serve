//! The log-mel front ends' shared pieces: the Slaney filterbank (librosa's and transformers' `norm="slaney"`,
//! `mel_scale="slaney"`) and torchaudio's centered reflect padding. Each model frames, floors and normalizes its own.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");

// Slaney mel scale: linear below 1 kHz, log above (torchaudio mel_scale="slaney").
fn hzToMelSlaney(f: f64) f64 {
    return if (f < 1000.0) 3.0 * f / 200.0 else 15.0 + 27.0 * @log(f / 1000.0) / @log(6.4);
}
fn melToHzSlaney(m: f64) f64 {
    return if (m < 15.0) 200.0 * m / 3.0 else 1000.0 * @exp((m - 15.0) * @log(6.4) / 27.0);
}

/// Slaney-scale, slaney-area-normalized mel filterbank `[n_mels, n_fft/2+1]` (f32) over 0..Nyquist,
/// librosa's and transformers' `mel_filter_bank(norm="slaney", mel_scale="slaney")`.
pub fn slaneyMelFilterbank(allocator: std.mem.Allocator, n_mels: usize, n_fft: usize, sample_rate: u32) !mlx.mlx_array {
    const n_freqs = n_fft / 2 + 1;
    const nyq: f64 = @as(f64, @floatFromInt(sample_rate)) / 2.0;
    const m_min = hzToMelSlaney(0.0);
    const m_max = hzToMelSlaney(nyq);
    const f_pts = try allocator.alloc(f64, n_mels + 2);
    defer allocator.free(f_pts);
    for (f_pts, 0..) |*f, i| {
        const m = m_min + (m_max - m_min) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_mels + 1));
        f.* = melToHzSlaney(m);
    }
    const fb = try allocator.alloc(f32, n_mels * n_freqs);
    defer allocator.free(fb);
    for (0..n_mels) |m| {
        const lo = f_pts[m];
        const ctr = f_pts[m + 1];
        const hi = f_pts[m + 2];
        const enorm = 2.0 / (hi - lo); // slaney area norm
        for (0..n_freqs) |f| {
            const fr = nyq * @as(f64, @floatFromInt(f)) / @as(f64, @floatFromInt(n_freqs - 1));
            const up = (fr - lo) / (ctr - lo);
            const down = (hi - fr) / (hi - ctr);
            var v = @min(up, down);
            if (v < 0) v = 0;
            fb[m * n_freqs + f] = @floatCast(v * enorm);
        }
    }
    const shape = [_]c_int{ @intCast(n_mels), @intCast(n_freqs) };
    return mlx.mlx_array_new_data(fb.ptr, &shape, 2, .float32);
}

/// Reflect-pad (mirror, no boundary repeat) — torchaudio center=True,
/// pad_mode="reflect". Needs `x.len > pad`. Caller frees.
pub fn reflectPadCenterF32(allocator: std.mem.Allocator, x: []const f32, pad: usize) ![]f32 {
    const out = try allocator.alloc(f32, x.len + 2 * pad);
    for (0..pad) |i| out[i] = x[pad - i];
    @memcpy(out[pad .. pad + x.len], x);
    for (0..pad) |i| out[pad + x.len + i] = x[x.len - 2 - i];
    return out;
}
