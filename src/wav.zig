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
