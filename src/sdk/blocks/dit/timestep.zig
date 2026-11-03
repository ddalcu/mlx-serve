//! The sinusoidal timestep features the DiTs feed their time MLP, computed on the host:
//! freq_i = exp(-ln(10000)·i/half), arg = t·freq_i, the cos half and the sin half in `order`,
//! in precision `T` and stored f32. f32: FLUX.2 (sin-first then flipped), Hunyuan3D paint
//! UNet, TRELLIS.2, Qwen-Image and ACE-Step (t×1000 passed in); f64: minimax_h3, Hunyuan3D
//! shape (sin first). mage_flow (bf16-rounded freqs, ×1000 after), MOSS-Image (`pow`),
//! Stable Audio 3 (learned exponential ramp) and LTX (on the GPU) compute other bits.

const std = @import("std");

pub const Order = enum { cos_first, sin_first };

pub fn sinusoid(comptime T: type, out: []f32, t: T, order: Order) void {
    const half = out.len / 2;
    const ln: T = @log(@as(T, 10000.0));
    for (0..half) |i| {
        const freq = @exp(-ln * @as(T, @floatFromInt(i)) / @as(T, @floatFromInt(half)));
        const arg = t * freq;
        const c: f32 = @floatCast(@cos(arg));
        const s: f32 = @floatCast(@sin(arg));
        out[i] = if (order == .cos_first) c else s;
        out[half + i] = if (order == .cos_first) s else c;
    }
}

// ── tests ──

const testing = std.testing;

/// flux.zig `Dit.timeEmbed`'s features (before the MLP).
fn fluxFeatures(t: f32) [256]f32 {
    var buf: [256]f32 = undefined;
    const half = 128;
    for (0..half) |i| {
        const freq = std.math.exp(-std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / @as(f32, half));
        const arg = t * freq;
        buf[i] = @sin(arg);
        buf[half + i] = @cos(arg);
    }
    var flipped: [256]f32 = undefined;
    @memcpy(flipped[0..half], buf[half..256]);
    @memcpy(flipped[half..256], buf[0..half]);
    return flipped;
}

/// hunyuan3d paint_unet.zig `timestepEmbed`.
fn paintUnetFeatures(out: []f32, t: f32) void {
    const half = out.len / 2;
    for (0..half) |i| {
        const freq = std.math.exp(-std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = t * freq;
        out[i] = std.math.cos(arg);
        out[half + i] = std.math.sin(arg);
    }
}

/// trellis2.zig `Dit.timeEmbedding`'s features.
fn trellisFeatures(buf: []f32, t: f32) void {
    const half = buf.len / 2;
    for (0..half) |i| {
        const freq = @exp(-@as(f32, @floatCast(@log(10000.0))) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = t * freq;
        buf[i] = @cos(arg);
        buf[half + i] = @sin(arg);
    }
}

/// qwen_image.zig `Dit.timeEmbed`'s first row.
fn qwenFeatures(buf: []f32, tv: f32) void {
    for (0..128) |k| {
        const freq: f32 = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(k)) / 128.0);
        const arg = tv * 1000.0 * freq;
        buf[k] = @cos(arg);
        buf[128 + k] = @sin(arg);
    }
}

/// acestep.zig `timestepEmbed`'s features.
fn acestepFeatures(sin_buf: []f32, t_val: f32) void {
    const half = 128;
    const ts = t_val * 1000.0;
    for (0..half) |i| {
        const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = ts * freq;
        sin_buf[i] = @cos(arg);
        sin_buf[half + i] = @sin(arg);
    }
}

/// minimax_h3 h3.zig `Model.timeEmbed`'s features for one t.
fn h3Features(buf: []f32, t: f64) void {
    const half = buf.len / 2;
    const log10k = @log(10000.0);
    for (0..half) |j| {
        const freq = @exp(-log10k * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(half)));
        const arg = t * freq;
        buf[j] = @floatCast(@cos(arg));
        buf[half + j] = @floatCast(@sin(arg));
    }
}

/// hunyuan3d.zig `timestepEmbed`'s buffer.
fn hunyuanFeatures(buf: []f32, t: f32) void {
    const half = buf.len / 2;
    const ln1e4: f64 = std.math.log(f64, std.math.e, 10000.0);
    for (0..half) |i| {
        const f = std.math.exp(-ln1e4 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half)));
        const ang = @as(f64, t) * f;
        buf[i] = @floatCast(@sin(ang));
        buf[half + i] = @floatCast(@cos(ang));
    }
}

test "media blocks: sinusoid equals every backend's timestep features bit for bit" {
    var got: [256]f32 = undefined;
    var want: [256]f32 = undefined;
    for ([_]f32{ 0, 1e-3, 0.37, 1, 17.5, 250, 999 }) |t| {
        sinusoid(f32, &got, t, .cos_first);
        try testing.expectEqualSlices(f32, &fluxFeatures(t), &got);
        paintUnetFeatures(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);
        trellisFeatures(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);

        sinusoid(f32, &got, t * 1000.0, .cos_first);
        qwenFeatures(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);
        acestepFeatures(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);

        sinusoid(f64, &got, t, .cos_first);
        h3Features(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);
        sinusoid(f64, &got, t, .sin_first);
        hunyuanFeatures(&want, t);
        try testing.expectEqualSlices(f32, &want, &got);
    }
}
