//! Host-side audio for Ming-3.0-tts: WAV loading, the torchaudio resampler,
//! the Kaldi 80-bin fbank CAM++ consumes, and trailing-silence trimming.
//! Pure Zig, no MLX.

const std = @import("std");
const wav_mod = @import("../../sdk/media/wav.zig");

/// A WAV's bytes as mono f32 (channel mean). 16/24-bit PCM and 32-bit float only.
pub fn decodeMono(a: std.mem.Allocator, bytes: []const u8) !struct { pcm: []f32, sample_rate: u32 } {
    const d = try wav_mod.decode(a, bytes);
    if (d.channels == 1) return .{ .pcm = d.pcm, .sample_rate = d.sample_rate };
    defer a.free(d.pcm);
    const n = d.pcm.len / d.channels;
    const out = try a.alloc(f32, n);
    for (0..n) |i| {
        var acc: f32 = 0;
        for (0..d.channels) |c| acc += d.pcm[i * d.channels + c];
        out[i] = acc / @as(f32, @floatFromInt(d.channels));
    }
    return .{ .pcm = out, .sample_rate = d.sample_rate };
}

/// torchaudio.transforms.Resample (sinc_interp_hann, lowpass_filter_width 6,
/// rolloff 0.99), as ported to NumPy in engine-mlx `audio.py`. Output length
/// ceil(new * len / orig). Caller frees.
pub fn resample(a: std.mem.Allocator, x: []const f32, orig_sr: u32, new_sr: u32) ![]f32 {
    if (orig_sr == new_sr) return a.dupe(f32, x);
    const g = std.math.gcd(orig_sr, new_sr);
    const orig: usize = orig_sr / g;
    const new: usize = new_sr / g;
    const base: f64 = @as(f64, @floatFromInt(@min(orig, new))) * 0.99;
    const width: usize = @intFromFloat(@ceil(6.0 * @as(f64, @floatFromInt(orig)) / base));
    const K = 2 * width + orig;
    const kernel = try a.alloc(f32, new * K);
    defer a.free(kernel);
    for (0..new) |i| {
        // (arange(0, -new, -1, f32) / new) in f32, then promoted to f64.
        const ti: f64 = @floatCast(-@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(new)));
        for (0..K) |k| {
            const idx = (@as(f64, @floatFromInt(k)) - @as(f64, @floatFromInt(width))) / @as(f64, @floatFromInt(orig));
            var t = std.math.clamp((ti + idx) * base, -6.0, 6.0);
            const w = std.math.pow(f64, @cos(t * std.math.pi / 6.0 / 2.0), 2);
            t *= std.math.pi;
            const sinc: f64 = if (t == 0) 1.0 else @sin(t) / t;
            kernel[i * K + k] = @floatCast(sinc * w * (base / @as(f64, @floatFromInt(orig))));
        }
    }
    const len = x.len;
    const lp = len + 2 * width + orig;
    const J = (lp - K) / orig + 1;
    const n_out: usize = (new * len + orig - 1) / orig;
    const out = try a.alloc(f32, n_out);
    for (0..J) |j| {
        for (0..new) |i| {
            const o = j * new + i;
            if (o >= n_out) break;
            var acc: f64 = 0;
            for (0..K) |k| {
                const p = j * orig + k; // index into the padded signal
                if (p < width or p >= width + len) continue;
                acc += @as(f64, x[p - width]) * @as(f64, kernel[i * K + k]);
            }
            out[o] = @floatCast(acc);
        }
    }
    return out;
}

// ── Kaldi fbank (torchaudio.compliance.kaldi.fbank / kaldi-native-fbank) ──
// 16 kHz, 25 ms povey frames every 10 ms, snip_edges, no dither, DC removal,
// preemphasis 0.97, 512-point power spectrum, 80 mel bins 20 Hz–8 kHz, log.

const FRAME_LEN = 400;
const FRAME_SHIFT = 160;
const NFFT = 512;
pub const NUM_MELS = 80;

fn melScale(f: f64) f64 {
    return 1127.0 * @log(1.0 + f / 700.0);
}

/// Returns [frames * 80] row-major (caller frees). Empty when < 400 samples.
pub fn fbank(a: std.mem.Allocator, x: []const f32) ![]f32 {
    if (x.len < FRAME_LEN) return a.alloc(f32, 0);
    const frames = 1 + (x.len - FRAME_LEN) / FRAME_SHIFT;
    const out = try a.alloc(f32, frames * NUM_MELS);
    // Mel weights over bins 0..255 (Kaldi drops the Nyquist bin).
    var banks: [NUM_MELS][NFFT / 2]f64 = undefined;
    const mel_lo = melScale(20.0);
    const mel_hi = melScale(8000.0);
    const delta = (mel_hi - mel_lo) / @as(f64, NUM_MELS + 1);
    for (0..NUM_MELS) |b| {
        const left = mel_lo + @as(f64, @floatFromInt(b)) * delta;
        const center = left + delta;
        const right = center + delta;
        for (0..NFFT / 2) |i| {
            const mel = melScale(@as(f64, @floatFromInt(i)) * 16000.0 / NFFT);
            banks[b][i] = if (mel > left and mel < right)
                (if (mel <= center) (mel - left) / (center - left) else (right - mel) / (right - center))
            else
                0;
        }
    }
    var win: [FRAME_LEN]f64 = undefined;
    for (0..FRAME_LEN) |i| {
        const c = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / (FRAME_LEN - 1));
        win[i] = std.math.pow(f64, c, 0.85);
    }
    var re: [NFFT]f64 = undefined;
    var im: [NFFT]f64 = undefined;
    for (0..frames) |fi| {
        const seg = x[fi * FRAME_SHIFT ..][0..FRAME_LEN];
        var mean: f64 = 0;
        for (seg) |v| mean += v;
        mean /= FRAME_LEN;
        var buf: [FRAME_LEN]f64 = undefined;
        for (0..FRAME_LEN) |i| buf[i] = @as(f64, seg[i]) - mean;
        var i: usize = FRAME_LEN - 1;
        while (i > 0) : (i -= 1) buf[i] -= 0.97 * buf[i - 1];
        buf[0] -= 0.97 * buf[0];
        @memset(&re, 0);
        @memset(&im, 0);
        for (0..FRAME_LEN) |k| re[k] = buf[k] * win[k];
        fft(&re, &im);
        var power: [NFFT / 2]f64 = undefined;
        for (0..NFFT / 2) |k| power[k] = re[k] * re[k] + im[k] * im[k];
        for (0..NUM_MELS) |b| {
            var e: f64 = 0;
            for (0..NFFT / 2) |k| e += banks[b][k] * power[k];
            out[fi * NUM_MELS + b] = @floatCast(@log(@max(e, std.math.floatEps(f32))));
        }
    }
    return out;
}

/// In-place radix-2 complex FFT.
fn fft(re: *[NFFT]f64, im: *[NFFT]f64) void {
    var j: usize = 0;
    for (1..NFFT) |i| {
        var bit: usize = NFFT >> 1;
        while (j & bit != 0) : (bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std.mem.swap(f64, &re[i], &re[j]);
            std.mem.swap(f64, &im[i], &im[j]);
        }
    }
    var len: usize = 2;
    while (len <= NFFT) : (len <<= 1) {
        const ang = -2.0 * std.math.pi / @as(f64, @floatFromInt(len));
        var i: usize = 0;
        while (i < NFFT) : (i += len) {
            for (0..len / 2) |k| {
                const wr = @cos(ang * @as(f64, @floatFromInt(k)));
                const wi = @sin(ang * @as(f64, @floatFromInt(k)));
                const ur = re[i + k];
                const ui = im[i + k];
                const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
                const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
            }
        }
    }
}

/// Target level for reference audio: the speech RMS of the the reference voice
/// presets (db30.wav -21.5, lingguang.wav -23.8 dBFS).
pub const PROMPT_SPEECH_RMS_DB: f32 = -22.0;
const PROMPT_PEAK_MAX: f32 = 0.89; // -1 dBFS

/// Scale reference audio in place so its speech RMS is PROMPT_SPEECH_RMS_DB,
/// with the peak capped at -1 dBFS. The model reproduces the reference's
/// level, so a quiet take (e.g. peak -23, RMS -42 dBFS) otherwise yields
/// equally quiet output. Speech RMS: 20 ms frames within 30 dB of the
/// loudest frame. Departs from the reference, which uses the reference as is.
/// Returns the gain applied.
pub fn normalizePrompt(x: []f32, sample_rate: u32) f32 {
    const n: usize = @max(1, sample_rate / 50);
    if (x.len < n) return 1;
    var max_ms: f64 = 0;
    var i: usize = 0;
    while (i + n <= x.len) : (i += n) max_ms = @max(max_ms, meanSquare(x[i..][0..n]));
    if (max_ms <= 1e-12) return 1;
    const floor = max_ms * 1e-3; // -30 dB in power
    var sum: f64 = 0;
    var count: usize = 0;
    i = 0;
    while (i + n <= x.len) : (i += n) {
        const ms = meanSquare(x[i..][0..n]);
        if (ms >= floor) {
            sum += ms;
            count += 1;
        }
    }
    const rms: f32 = @floatCast(@sqrt(sum / @as(f64, @floatFromInt(count))));
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    const gain = @min(std.math.pow(f32, 10, PROMPT_SPEECH_RMS_DB / 20) / rms, PROMPT_PEAK_MAX / peak);
    for (x) |*v| v.* *= gain;
    return gain;
}

fn meanSquare(x: []const f32) f64 {
    var s: f64 = 0;
    for (x) |v| s += @as(f64, v) * v;
    return s / @as(f64, @floatFromInt(x.len));
}

/// Soft knee above 0.9: |x| > 0.9 maps to 0.9 + 0.08·tanh((|x| - 0.9) / 0.08),
/// so output stays under 0.98 (-0.2 dBFS) and peaks round off instead of
/// clipping. Plain TTS (no reference) often peaks above 1.0, which the 16-bit
/// WAV would otherwise hard-clip. Per sample, so streamed chunks match the
/// full output. Departs from the reference, which writes the samples as is.
pub fn softLimit(x: []f32) void {
    for (x) |*v| {
        const m = @abs(v.*);
        if (m > 0.9) v.* = std.math.copysign(0.9 + 0.08 * std.math.tanh((m - 0.9) / 0.08), v.*);
    }
}

/// Vendor `ming_tts.trim_trailing_silence`: cut after the last 100 ms frame
/// whose mean |x| exceeds 1e-3, keeping 300 ms of tail. Returns a sub-slice.
pub fn trimTrailingSilence(x: []const f32, sample_rate: u32) []const f32 {
    const frame: usize = sample_rate / 10;
    const tail: usize = @intFromFloat(@as(f64, @floatFromInt(sample_rate)) * 0.3);
    if (x.len == 0) return x;
    if (x.len < frame) return x[0..@min(tail, x.len)];
    const n = (x.len - frame) / frame + 1;
    const body = x[0 .. n * frame];
    var idx: isize = @as(isize, @intCast(n)) - 1;
    while (idx >= 0) : (idx -= 1) {
        const seg = body[@as(usize, @intCast(idx)) * frame ..][0..frame];
        var s: f64 = 0;
        for (seg) |v| s += @abs(v);
        if (s / @as(f64, @floatFromInt(frame)) > 1e-3) break;
    }
    if (idx < 0) return body[0..@min(tail, body.len)];
    const end = @as(usize, @intCast(idx)) * frame + frame + tail;
    return body[0..@min(end, body.len)];
}

/// Sample range from the first to the end of the last 10 ms frame whose RMS
/// is within `SPEECH_RANGE_DB` of the loudest frame and above
/// `SPEECH_FLOOR_DB`. Empty (0, 0) when no frame qualifies.
pub fn speechBounds(x: []const f32, sample_rate: u32) struct { start: usize, end: usize } {
    const n: usize = @max(1, sample_rate / 100);
    const frames = x.len / n;
    var max_ms: f64 = 0;
    for (0..frames) |k| max_ms = @max(max_ms, meanSquare(x[k * n ..][0..n]));
    const thr = @max(max_ms * std.math.pow(f64, 10, -SPEECH_RANGE_DB / 10), std.math.pow(f64, 10, SPEECH_FLOOR_DB / 10));
    var first: ?usize = null;
    var last: usize = 0;
    for (0..frames) |k| if (meanSquare(x[k * n ..][0..n]) >= thr) {
        if (first == null) first = k;
        last = k;
    };
    const f = first orelse return .{ .start = 0, .end = 0 };
    return .{ .start = f * n, .end = (last + 1) * n };
}

pub const SPEECH_RANGE_DB: f64 = 45;
pub const SPEECH_FLOOR_DB: f64 = -60;

/// Speech RMS in dBFS: mean power of the 20 ms frames within 30 dB of the
/// loudest one (the measure `normalizePrompt` uses). -inf for silence.
pub fn speechRmsDb(x: []const f32, sample_rate: u32) f64 {
    const n: usize = @max(1, sample_rate / 50);
    var max_ms: f64 = 0;
    var i: usize = 0;
    while (i + n <= x.len) : (i += n) max_ms = @max(max_ms, meanSquare(x[i..][0..n]));
    if (max_ms <= 1e-12) return -std.math.inf(f64);
    var sum: f64 = 0;
    var count: usize = 0;
    i = 0;
    while (i + n <= x.len) : (i += n) {
        const ms = meanSquare(x[i..][0..n]);
        if (ms >= max_ms * 1e-3) {
            sum += ms;
            count += 1;
        }
    }
    return 10 * std.math.log10(sum / @as(f64, @floatFromInt(count)));
}

/// Raised-cosine fade over the first (`in`) or last `n` samples of `x`.
pub fn fade(x: []f32, n_in: usize, dir: enum { in, out }) void {
    const n = @min(n_in, x.len);
    for (0..n) |k| {
        const w: f32 = @floatCast(0.5 - 0.5 * @cos(std.math.pi * (@as(f64, @floatFromInt(k)) + 0.5) / @as(f64, @floatFromInt(n))));
        switch (dir) {
            .in => x[k] *= w,
            .out => x[x.len - 1 - k] *= w,
        }
    }
}

test "ming_tts audio: speechBounds finds the loud span" {
    var x: [44100]f32 = @splat(0);
    for (4410..8820) |i| x[i] = 0.3 * @sin(@as(f32, @floatFromInt(i)) * 0.2);
    for (0..44100) |i| x[i] += if (i % 2 == 0) 1e-5 else -1e-5; // noise floor
    const b = speechBounds(&x, 44100);
    try std.testing.expectEqual(@as(usize, 4410), b.start);
    try std.testing.expectEqual(@as(usize, 8820), b.end);
    var y: [100]f32 = @splat(1);
    fade(&y, 10, .in);
    fade(&y, 10, .out);
    try std.testing.expect(y[0] < 0.05 and y[99] < 0.05 and y[50] == 1);
}

test "ming_tts audio: resample keeps a DC level and the torchaudio output length" {
    const a = std.testing.allocator;
    var x: [4800]f32 = undefined;
    @memset(&x, 0.5);
    const y = try resample(a, &x, 48000, 16000);
    defer a.free(y);
    try std.testing.expectEqual(@as(usize, 1600), y.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y[800], 1e-3);
}

test "ming_tts audio: normalizePrompt lifts a quiet clip to the target speech level" {
    var x: [16000]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = 0.01 * @sin(@as(f32, @floatFromInt(i)) * 0.1);
    const gain = normalizePrompt(&x, 16000);
    try std.testing.expect(gain > 1);
    var s: f64 = 0;
    for (x) |v| s += @as(f64, v) * v;
    const db = 10 * std.math.log10(s / x.len);
    try std.testing.expectApproxEqAbs(@as(f64, PROMPT_SPEECH_RMS_DB), db, 0.1);
}

test "ming_tts audio: trim keeps 300 ms after the last loud frame" {
    var x: [44100]f32 = @splat(0);
    for (0..4410) |i| x[i] = 0.5;
    const t = trimTrailingSilence(&x, 44100);
    try std.testing.expectEqual(@as(usize, 4410 + 13230), t.len);
}
