//! The LFM3 speech encoder and projector, mlx-vlm's `lfm3_audio/audio.py`: a 16 kHz clip's log-mel (128 Slaney bins
//! every 10 ms, uncentered frames, floored at -12, `(x + 4) / 4`), three stride-2 convs over the (mel, time) plane
//! into one vector per 8 frames, a 17-layer LFM2 trunk (`trunk.zig`: gated short convs, attention over the last
//! `window` rows),
//! then LayerNorm and a GELU MLP to the text width. Activations run in the weights' dtype, SiLU and GELU in f32.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const mel_bank = @import("../../sdk/media/mel.zig");
const config = @import("config.zig");
const trunk_mod = @import("trunk.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

pub const SAMPLE_RATE = 16000;
const N_FFT = 400;
const HOP = 160;
const LOG_FLOOR = -12.0;

/// Feature frames of a clip of `samples`: it is right-padded to `400 + k * 160` samples.
pub fn frames(samples: usize) usize {
    return 1 + (if (samples > N_FFT) (samples - N_FFT + HOP - 1) / HOP else 0);
}

/// Encoder rows of `n` feature frames: three stride-2 convs, `ceil(n / 8)`.
pub fn rows(n: usize) usize {
    return (n + 7) / 8;
}

/// The clip right-padded with zeros to its frames' span (caller frees).
pub fn padded(a: std.mem.Allocator, clip: []const f32) ![]f32 {
    const out = try a.alloc(f32, N_FFT + (frames(clip.len) - 1) * HOP);
    @memcpy(out[0..clip.len], clip);
    @memset(out[clip.len..], 0);
    return out;
}

const Conv = struct { w: Array, b: Array };

pub const Tower = struct {
    cfg: config.Audio,
    /// The Slaney filterbank `[128, 201]` and the periodic Hann window `[400]`, f32.
    filters: Array,
    window: Array,
    conv_in: Conv,
    conv_mid: Conv,
    lin_mid: Conv,
    conv_end: Conv,
    lin_end: Conv,
    lin_out: Linear,
    trunk: trunk_mod.Trunk,
    proj_norm_w: Array,
    proj_norm_b: Array,
    linear_1: Linear,
    linear_2: Linear,
    s: mlx.mlx_stream,

    /// The encoder's and the projector's tensors, owned by `owned`; null when the pack ships no encoder.
    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Audio, block_json: []const u8, owned: *reader.Owned) !?Tower {
        var r = try reader.Reader.open(ctx, block_json, "audio_tower");
        defer r.close();
        if (!r.has("audio_tower.downsample.conv_in.weight")) return null;
        const filters = try mel_bank.slaneyMelFilterbank(gpa, cfg.mel_bins, N_FFT, SAMPLE_RATE);
        try owned.own(filters);
        var hann: [N_FFT]f32 = undefined;
        for (&hann, 0..) |*w, i| w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / N_FFT));
        const window = mlx.mlx_array_new_data(&hann, &[_]c_int{N_FFT}, 1, .float32);
        try owned.own(window);
        const h = cfg.hidden;
        var t: Tower = .{
            .cfg = cfg,
            .filters = filters,
            .window = window,
            .conv_in = try conv(&r, owned, "audio_tower.downsample.conv_in"),
            .conv_mid = try conv(&r, owned, "audio_tower.downsample.conv_mid"),
            .lin_mid = try conv(&r, owned, "audio_tower.downsample.lin_mid"),
            .conv_end = try conv(&r, owned, "audio_tower.downsample.conv_end"),
            .lin_end = try conv(&r, owned, "audio_tower.downsample.lin_end"),
            .lin_out = try biased(&r, owned, "audio_tower.downsample.lin_out", cfg.conv * cfg.mel_bins / 8),
            .trunk = try trunk_mod.Trunk.load(&r, owned, "audio_tower.lfm", cfg.attention[0..cfg.layers], h, cfg.eps, cfg.theta, .{ .causal = cfg.window }),
            .proj_norm_w = try arr(&r, owned, "audio_projector.layer_norm.weight"),
            .proj_norm_b = try arr(&r, owned, "audio_projector.layer_norm.bias"),
            .linear_1 = try biased(&r, owned, "audio_projector.linear_1", h),
            .linear_2 = undefined,
            .s = r.s,
        };
        errdefer t.trunk.deinit();
        t.linear_2 = try biased(&r, owned, "audio_projector.linear_2", @intCast(ops.shape(t.linear_1.w)[0]));
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.trunk.deinit();
    }

    /// Log-mel `[128, n_frames]` of a clip padded by `padded` (f32).
    pub fn features(t: *const Tower, g: *Graph, wave: Array, n_frames: usize) !Array {
        const framed = try g.call(mlx.mlx_as_strided, .{ wave, &[_]c_int{ @intCast(n_frames), N_FFT }, @as(usize, 2), &[_]i64{ HOP, 1 }, @as(usize, 2), @as(usize, 0) });
        const spec = try g.call(mlx.mlx_fft_rfft, .{ try g.mul(framed, t.window), @as(c_int, N_FFT), @as(c_int, 1), mlx.MLX_FFT_NORM_BACKWARD });
        const power = try g.square(try g.abs(spec));
        const mel = try g.matmul(t.filters, try g.transpose(power, &.{ 1, 0 }));
        const log10 = try g.mulScalar(try g.log(mel), 1.0 / @log(10.0));
        return g.mulScalar(try g.addScalar(try g.maximum(log10, try g.scalar(LOG_FLOOR, .float32)), 4.0), 0.25);
    }

    /// The clip's rows `[rows(n_frames), text_hidden]` (caller owns) from its padded samples.
    pub fn forward(t: *const Tower, wave: Array, n_frames: usize) !mlx.mlx_array {
        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const feats = try t.features(&g, wave, n_frames);
        const enc = try t.trunk.forward(&g, try t.downsample(&g, feats), 0);
        const n: c_int = ops.shape(enc)[1];
        const normed = try g.layerNorm(enc, t.proj_norm_w, t.proj_norm_b, 1e-5);
        const h1 = try t.linear_1.forward(&g, normed);
        const out = try t.linear_2.forward(&g, try f32Act(&g, h1, .gelu));
        return g.escape(try g.reshape(out, &.{ n, -1 }));
    }

    /// `[128, T]` features over the (mel, time) plane as NHWC: three stride-2 3x3 convs (the last two depthwise, each
    /// with a pointwise mix), then (channel, mel) channel-major into `lin_out`: `[1, ceil(T / 8), hidden]`.
    fn downsample(t: *const Tower, g: *Graph, feats: Array) !Array {
        const sh = ops.shape(feats);
        const dt = mlx.mlx_array_dtype(.{ .ctx = t.conv_in.w.ctx });
        var x = try g.astype(try g.reshape(feats, &.{ 1, sh[0], sh[1], 1 }), dt);
        x = try f32Act(g, try convAt(g, x, t.conv_in, 2, 1, 1), .silu);
        const c: c_int = @intCast(t.cfg.conv);
        x = try f32Act(g, try convAt(g, try convAt(g, x, t.conv_mid, 2, 1, c), t.lin_mid, 1, 0, 1), .silu);
        x = try f32Act(g, try convAt(g, try convAt(g, x, t.conv_end, 2, 1, c), t.lin_end, 1, 0, 1), .silu);
        const s4 = ops.shape(x);
        const flat = try g.reshape(try g.transpose(x, &.{ 0, 2, 3, 1 }), &.{ 1, s4[2], s4[3] * s4[1] });
        return t.lin_out.forward(g, flat);
    }
};

fn convAt(g: *Graph, x: Array, c: Conv, stride: c_int, pad: c_int, groups: c_int) !Array {
    return g.add(try g.conv2d(x, c.w, stride, pad, 1, groups), c.b);
}

const Act = enum { silu, gelu };

/// The activation in f32, rounded once to `x`'s dtype (the reference evaluates bf16 activations in f32).
fn f32Act(g: *Graph, x: Array, comptime act: Act) !Array {
    const dt = mlx.mlx_array_dtype(.{ .ctx = x.ctx });
    const f = try g.astype(x, .float32);
    const y = switch (act) {
        .silu => try g.silu(f),
        .gelu => try g.mul(try g.mulScalar(f, 0.5), try g.addScalar(try g.erf(try g.mulScalar(f, std.math.sqrt1_2)), 1.0)),
    };
    return g.astype(y, dt);
}

fn conv(r: *reader.Reader, owned: *reader.Owned, base: []const u8) !Conv {
    var nb: [160]u8 = undefined;
    return .{
        .w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.weight", .{base})),
        .b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.bias", .{base})),
    };
}

fn lin(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.proj(base, in_dim);
    try owned.linear(l);
    return l;
}

fn biased(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    var l = try lin(r, owned, base, in_dim);
    var nb: [160]u8 = undefined;
    l.bias = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    return l;
}

fn arr(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const a = try r.need(full);
    try owned.own(a);
    return a;
}

// ── tests ──

test "lfm3 audio: a clip pads to 400 + k*160 samples, one row per 8 frames" {
    try std.testing.expectEqual(@as(usize, 1), frames(1));
    try std.testing.expectEqual(@as(usize, 1), frames(400));
    try std.testing.expectEqual(@as(usize, 2), frames(401));
    try std.testing.expectEqual(@as(usize, 49), frames(8000));
    try std.testing.expectEqual(@as(usize, 7), rows(49));
    const ones: [401]f32 = @splat(1);
    const p = try padded(std.testing.allocator, &ones);
    defer std.testing.allocator.free(p);
    try std.testing.expectEqual(@as(usize, 560), p.len);
}
