//! Minimal WAV (RIFF/WAVE) encoder for the native audio-generation path.
//!
//! TTS produces a mono float waveform in [-1, 1] at a model sample rate (24 kHz
//! for Qwen3-TTS). We serialize it as a standard 16-bit PCM WAV — the most
//! widely compatible container, and all the app needs to play it back. Float
//! equivalence against the Python reference is checked on the raw samples (the
//! `.npy` oracle), not on these quantized bytes, so 16-bit PCM here is fine.
//!
//! No external dependency: a WAV file is a 44-byte header followed by the PCM
//! payload. This module is pure (allocator + slice in, owned byte slice out) so
//! it is trivially unit-testable.

const std = @import("std");

/// Encode `samples` (mono, f32 in [-1, 1]) as a 16-bit PCM WAV at `sample_rate`.
/// Returns an owned byte slice the caller frees. Samples are clamped before
/// quantization so out-of-range values can't wrap.
pub fn encodePcm16Mono(
    allocator: std.mem.Allocator,
    samples: []const f32,
    sample_rate: u32,
) ![]u8 {
    return encodePcm16(allocator, samples, sample_rate, 1);
}

/// Encode interleaved `samples` (f32 in [-1, 1]) as 16-bit PCM WAV with
/// `channels` channels at `sample_rate`. `samples.len` must be a multiple of
/// `channels`.
pub fn encodePcm16(
    allocator: std.mem.Allocator,
    samples: []const f32,
    sample_rate: u32,
    channels: u16,
) ![]u8 {
    std.debug.assert(channels >= 1);
    std.debug.assert(samples.len % channels == 0);

    const bits_per_sample: u16 = 16;
    const bytes_per_sample: u32 = bits_per_sample / 8;
    const block_align: u16 = @intCast(channels * bytes_per_sample);
    const byte_rate: u32 = sample_rate * block_align;
    const data_bytes: u32 = @intCast(samples.len * bytes_per_sample);
    const total: usize = 44 + data_bytes;

    var buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);

    // RIFF header
    @memcpy(buf[0..4], "RIFF");
    writeU32LE(buf[4..8], @intCast(36 + data_bytes)); // file size - 8
    @memcpy(buf[8..12], "WAVE");

    // fmt subchunk (PCM)
    @memcpy(buf[12..16], "fmt ");
    writeU32LE(buf[16..20], 16); // PCM fmt chunk size
    writeU16LE(buf[20..22], 1); // audio format = PCM
    writeU16LE(buf[22..24], channels);
    writeU32LE(buf[24..28], sample_rate);
    writeU32LE(buf[28..32], byte_rate);
    writeU16LE(buf[32..34], block_align);
    writeU16LE(buf[34..36], bits_per_sample);

    // data subchunk
    @memcpy(buf[36..40], "data");
    writeU32LE(buf[40..44], data_bytes);

    var off: usize = 44;
    for (samples) |s| {
        const clamped = std.math.clamp(s, -1.0, 1.0);
        // Symmetric scale by 32767 (matches common float→PCM16 conventions and
        // avoids -32768 overflow on exactly -1.0 after rounding).
        const scaled = std.math.round(clamped * 32767.0);
        const v: i16 = @intFromFloat(scaled);
        writeI16LE(buf[off .. off + 2], v);
        off += 2;
    }

    return buf;
}

/// A decoded WAV: interleaved f32 samples in [-1, 1] plus the file's real
/// format metadata (unlike the legacy mono-PCM16 fast path in gen.zig, which
/// assumes the app pre-normalized the clip).
pub const Decoded = struct {
    pcm: []f32,
    channels: u16,
    sample_rate: u32,
};

/// Decode a RIFF/WAVE file. Parses the `fmt ` chunk — 16/24-bit PCM and
/// 32-bit IEEE float, including WAVE_FORMAT_EXTENSIBLE — and the `data`
/// chunk (scanning past extra chunks like `LIST`). Returns interleaved f32.
pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) !Decoded {
    if (bytes.len < 44 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.BadWav;

    var format: u16 = 0;
    var channels: u16 = 0;
    var sample_rate: u32 = 0;
    var bits: u16 = 0;
    var data: ?[]const u8 = null;

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const cid = bytes[pos .. pos + 4];
        const csize: usize = std.mem.readInt(u32, bytes[pos + 4 ..][0..4], .little);
        const body_start = pos + 8;
        const body_end = @min(body_start + csize, bytes.len);
        if (std.mem.eql(u8, cid, "fmt ") and body_end - body_start >= 16) {
            const f = bytes[body_start..body_end];
            format = std.mem.readInt(u16, f[0..2], .little);
            channels = std.mem.readInt(u16, f[2..4], .little);
            sample_rate = std.mem.readInt(u32, f[4..8], .little);
            bits = std.mem.readInt(u16, f[14..16], .little);
            // WAVE_FORMAT_EXTENSIBLE: the effective codec is the sub-format
            // GUID's leading u16 (offset 24 into the fmt body).
            if (format == 0xFFFE and f.len >= 26) {
                format = std.mem.readInt(u16, f[24..26], .little);
            }
        } else if (std.mem.eql(u8, cid, "data")) {
            data = bytes[body_start..body_end];
        }
        pos = body_end + (csize & 1); // chunks are word-aligned
    }

    const payload = data orelse return error.NoDataChunk;
    if (channels == 0 or sample_rate == 0) return error.BadWav;

    const pcm: []f32 = switch (format) {
        1 => switch (bits) {
            16 => blk: {
                const n = payload.len / 2;
                const out = try allocator.alloc(f32, n);
                for (0..n) |i| {
                    const v = std.mem.readInt(i16, payload[i * 2 ..][0..2], .little);
                    out[i] = @as(f32, @floatFromInt(v)) / 32768.0;
                }
                break :blk out;
            },
            24 => blk: {
                const n = payload.len / 3;
                const out = try allocator.alloc(f32, n);
                for (0..n) |i| {
                    const b = payload[i * 3 ..][0..3];
                    const raw: i32 = @as(i32, b[0]) | (@as(i32, b[1]) << 8) | (@as(i32, @as(i8, @bitCast(b[2]))) << 16);
                    out[i] = @as(f32, @floatFromInt(raw)) / 8388608.0;
                }
                break :blk out;
            },
            else => return error.UnsupportedWavFormat,
        },
        3 => blk: {
            if (bits != 32) return error.UnsupportedWavFormat;
            const n = payload.len / 4;
            const out = try allocator.alloc(f32, n);
            for (0..n) |i| {
                const u = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
                out[i] = @bitCast(u);
            }
            break :blk out;
        },
        else => return error.UnsupportedWavFormat,
    };

    return .{ .pcm = pcm, .channels = channels, .sample_rate = sample_rate };
}

/// Linear-resample interleaved audio from `from_rate` to `to_rate`. Enough
/// for conditioning inputs (the muxed output keeps the original samples, so
/// resampler quality never reaches the user's ears).
pub fn resampleLinear(
    allocator: std.mem.Allocator,
    pcm: []const f32,
    channels: u16,
    from_rate: u32,
    to_rate: u32,
) ![]f32 {
    std.debug.assert(channels >= 1);
    if (from_rate == to_rate) return allocator.dupe(f32, pcm);
    const ch: usize = channels;
    const frames_in = pcm.len / ch;
    if (frames_in == 0) return allocator.alloc(f32, 0);
    const frames_out: usize = @intFromFloat(@floor(
        @as(f64, @floatFromInt(frames_in)) * @as(f64, @floatFromInt(to_rate)) / @as(f64, @floatFromInt(from_rate)),
    ));
    const out = try allocator.alloc(f32, @max(frames_out, 1) * ch);
    const ratio = @as(f64, @floatFromInt(from_rate)) / @as(f64, @floatFromInt(to_rate));
    for (0..@max(frames_out, 1)) |i| {
        const src = @as(f64, @floatFromInt(i)) * ratio;
        const f0: usize = @min(@as(usize, @intFromFloat(@floor(src))), frames_in - 1);
        const f1: usize = @min(f0 + 1, frames_in - 1);
        const frac: f32 = @floatCast(src - @floor(src));
        for (0..ch) |c| {
            out[i * ch + c] = pcm[f0 * ch + c] * (1.0 - frac) + pcm[f1 * ch + c] * frac;
        }
    }
    return out;
}

/// Force interleaved audio to stereo: mono duplicates into both channels,
/// stereo copies, >2 channels keeps the first two (front L/R by convention).
pub fn toStereoInterleaved(allocator: std.mem.Allocator, pcm: []const f32, channels: u16) ![]f32 {
    std.debug.assert(channels >= 1);
    const ch: usize = channels;
    const frames = pcm.len / ch;
    const out = try allocator.alloc(f32, frames * 2);
    for (0..frames) |i| {
        out[i * 2] = pcm[i * ch];
        out[i * 2 + 1] = if (ch >= 2) pcm[i * ch + 1] else pcm[i * ch];
    }
    return out;
}

inline fn writeU16LE(dst: []u8, v: u16) void {
    std.mem.writeInt(u16, dst[0..2], v, .little);
}
inline fn writeU32LE(dst: []u8, v: u32) void {
    std.mem.writeInt(u32, dst[0..4], v, .little);
}
inline fn writeI16LE(dst: []u8, v: i16) void {
    std.mem.writeInt(i16, dst[0..2], v, .little);
}

// ── Tests ──

test "encodePcm16Mono header is a valid 24kHz mono PCM16 WAV" {
    const a = std.testing.allocator;
    const samples = [_]f32{ 0.0, 0.5, -0.5, 1.0, -1.0 };
    const wav = try encodePcm16Mono(a, &samples, 24000);
    defer a.free(wav);

    try std.testing.expectEqual(@as(usize, 44 + samples.len * 2), wav.len);
    try std.testing.expectEqualSlices(u8, "RIFF", wav[0..4]);
    try std.testing.expectEqualSlices(u8, "WAVE", wav[8..12]);
    try std.testing.expectEqualSlices(u8, "fmt ", wav[12..16]);
    try std.testing.expectEqualSlices(u8, "data", wav[36..40]);

    // audio format = 1 (PCM), channels = 1, sample rate = 24000
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, wav[20..22], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, wav[22..24], .little));
    try std.testing.expectEqual(@as(u32, 24000), std.mem.readInt(u32, wav[24..28], .little));
    // byte rate = 24000 * 1 * 2, block align = 2, bits = 16
    try std.testing.expectEqual(@as(u32, 48000), std.mem.readInt(u32, wav[28..32], .little));
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, wav[32..34], .little));
    try std.testing.expectEqual(@as(u16, 16), std.mem.readInt(u16, wav[34..36], .little));
    // data size
    try std.testing.expectEqual(@as(u32, @intCast(samples.len * 2)), std.mem.readInt(u32, wav[40..44], .little));
}

test "encodePcm16 quantizes and clamps samples correctly" {
    const a = std.testing.allocator;
    const samples = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0 }; // last two out of range → clamp
    const wav = try encodePcm16Mono(a, &samples, 16000);
    defer a.free(wav);

    const pcm = wav[44..];
    try std.testing.expectEqual(@as(i16, 0), std.mem.readInt(i16, pcm[0..2], .little));
    try std.testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, pcm[2..4], .little));
    try std.testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, pcm[4..6], .little));
    try std.testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, pcm[6..8], .little)); // clamped 2.0
    try std.testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, pcm[8..10], .little)); // clamped -2.0
}

test "encodePcm16 stereo block align and length" {
    const a = std.testing.allocator;
    const samples = [_]f32{ 0.1, 0.2, 0.3, 0.4 }; // 2 frames × 2 ch
    const wav = try encodePcm16(a, &samples, 48000, 2);
    defer a.free(wav);
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, wav[22..24], .little)); // channels
    try std.testing.expectEqual(@as(u16, 4), std.mem.readInt(u16, wav[32..34], .little)); // block align 2ch*2B
    try std.testing.expectEqual(@as(usize, 44 + 8), wav.len);
}

test "decode parses fmt (rate/channels) and PCM16 stereo data round-trip" {
    const a = std.testing.allocator;
    const samples = [_]f32{ 0.0, 0.25, -0.5, 0.75, 1.0, -1.0 }; // 3 frames × 2 ch
    const wav = try encodePcm16(a, &samples, 44100, 2);
    defer a.free(wav);

    const dec = try decode(a, wav);
    defer a.free(dec.pcm);
    try std.testing.expectEqual(@as(u32, 44100), dec.sample_rate);
    try std.testing.expectEqual(@as(u16, 2), dec.channels);
    try std.testing.expectEqual(samples.len, dec.pcm.len);
    for (samples, dec.pcm) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1.0 / 32000.0);
    }
}

test "decode reads IEEE float32 WAV" {
    const a = std.testing.allocator;
    // Hand-built minimal float WAV: fmt (format 3, 1 ch, 16 kHz, 32-bit) + data.
    const samples = [_]f32{ 0.5, -0.25, 1.0 };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.appendSlice(a, "RIFF\x00\x00\x00\x00WAVE");
    try buf.appendSlice(a, "fmt \x10\x00\x00\x00"); // fmt, size 16
    try buf.appendSlice(a, &.{ 3, 0, 1, 0 }); // format 3 (IEEE float), 1 channel
    try buf.appendSlice(a, &.{ 0x80, 0x3e, 0, 0 }); // 16000 Hz
    try buf.appendSlice(a, &.{ 0, 0xfa, 0, 0 }); // byte rate 64000
    try buf.appendSlice(a, &.{ 4, 0, 32, 0 }); // block align 4, bits 32
    try buf.appendSlice(a, "data\x0c\x00\x00\x00"); // 12 bytes
    for (samples) |s| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(s), .little);
        try buf.appendSlice(a, &b);
    }

    const dec = try decode(a, buf.items);
    defer a.free(dec.pcm);
    try std.testing.expectEqual(@as(u32, 16000), dec.sample_rate);
    try std.testing.expectEqual(@as(u16, 1), dec.channels);
    try std.testing.expectEqualSlices(f32, &samples, dec.pcm);
}

test "decode rejects unsupported codec (mu-law)" {
    const a = std.testing.allocator;
    const samples = [_]f32{0.0};
    const wav = try encodePcm16Mono(a, &samples, 8000);
    defer a.free(wav);
    var bad = try a.dupe(u8, wav);
    defer a.free(bad);
    std.mem.writeInt(u16, bad[20..22], 7, .little); // mu-law
    try std.testing.expectError(error.UnsupportedWavFormat, decode(a, bad));
}

test "resampleLinear 2:1 on a stereo ramp keeps every other frame" {
    const a = std.testing.allocator;
    // 4 frames × 2 ch at 32 kHz; halving picks frames 0 and 2 exactly.
    const pcm = [_]f32{ 0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0 };
    const out = try resampleLinear(a, &pcm, 2, 32000, 16000);
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, 4), out.len); // 2 frames × 2 ch
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), out[3], 1e-6);
}

test "resampleLinear same-rate is a copy" {
    const a = std.testing.allocator;
    const pcm = [_]f32{ 0.1, 0.2, 0.3 };
    const out = try resampleLinear(a, &pcm, 1, 16000, 16000);
    defer a.free(out);
    try std.testing.expectEqualSlices(f32, &pcm, out);
}

test "toStereoInterleaved duplicates mono, keeps stereo, drops extra channels" {
    const a = std.testing.allocator;
    const mono = [_]f32{ 0.1, 0.2 };
    const st = try toStereoInterleaved(a, &mono, 1);
    defer a.free(st);
    try std.testing.expectEqualSlices(f32, &.{ 0.1, 0.1, 0.2, 0.2 }, st);

    const stereo = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const st2 = try toStereoInterleaved(a, &stereo, 2);
    defer a.free(st2);
    try std.testing.expectEqualSlices(f32, &stereo, st2);

    const five_one = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // 2 frames × 6 ch
    const st3 = try toStereoInterleaved(a, &five_one, 6);
    defer a.free(st3);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 7, 8 }, st3);
}
