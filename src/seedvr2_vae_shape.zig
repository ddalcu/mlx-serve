//! SeedVR2 video-VAE geometry: the down/up ladder, causal head replication, and
//! the latent<->pixel shape contract. Pure integer math, zero MLX, so the part
//! of the VAE most likely to be silently wrong can be pinned before a single
//! weight is loaded.
//!
//! Transcribed from ByteDance-Seed/SeedVR `models/video_vae_v3/*` (Apache-2.0);
//! spec + trap list in `docs/seedvr2-arch.md` §3.
//!
//! WHY THIS IS ITS OWN FILE: the two strides in this VAE are governed by two
//! DIFFERENT predicates that disagree on block 3 — it is FLAGGED for temporal
//! downsampling and has no downsampler at all — so a port that collapses them
//! into one loop condition gets /8 in time instead of /4 and still produces a
//! plausible video. A parity test on the finished VAE reports one cosine
//! number and cannot tell you which of the two moved it.
//!
//! Every geometry claim below is now checked against the real
//! `ema_vae_fp16.safetensors` key list, and one of them was wrong when it was
//! only checked against the reference source — see `resnetConvGeom`.

const std = @import("std");

/// The 3B checkpoint's `s8_c16_t4_inflation_sd3` geometry.
pub const Config = struct {
    block_out_channels: []const u32 = &.{ 128, 256, 512, 512 },
    layers_per_block: u32 = 2,
    latent_channels: u32 = 16,
    norm_num_groups: u32 = 32,
    /// How many of the trailing blocks are FLAGGED for temporal striding. Note
    /// this is not the number that actually stride — see `temporalDownBlocks`.
    temporal_down_num: u32 = 2,
    /// `(raw - shift) * scale`; the checkpoint ships no shift.
    scaling_factor: f32 = 0.9152,

    pub fn numBlocks(self: Config) u32 {
        return @intCast(self.block_out_channels.len);
    }
};

/// True when encoder block `i` carries a downsampler at all. The final block
/// never does — this is the guard that makes block 3's temporal flag dead.
pub fn addDownsample(cfg: Config, i: u32) bool {
    return i != cfg.numBlocks() - 1;
}

/// True when encoder block `i` is FLAGGED for temporal striding.
/// `i >= len(block_out_channels) - temporal_down_num - 1`.
///
/// This is deliberately reported separately from whether the block actually
/// strides: the flag is set on the last block too, where it does nothing. A
/// port that collapses the two predicates into one loop condition gets /8 in
/// time instead of /4 and still runs.
pub fn temporalDownFlagged(cfg: Config, i: u32) bool {
    const n = cfg.numBlocks();
    // The reference computes `i >= n - temporal_down_num - 1` in Python ints,
    // where the right side can go negative; clamp rather than underflow.
    if (cfg.temporal_down_num + 1 >= n) return true;
    return i >= n - cfg.temporal_down_num - 1;
}

/// True when block `i` actually strides in time: flagged AND has a downsampler.
pub fn temporalDownEffective(cfg: Config, i: u32) bool {
    return temporalDownFlagged(cfg, i) and addDownsample(cfg, i);
}

/// Total spatial downsample factor implied by the ladder (expect 8).
pub fn spatialFactor(cfg: Config) u32 {
    var f: u32 = 1;
    var i: u32 = 0;
    while (i < cfg.numBlocks()) : (i += 1) {
        if (addDownsample(cfg, i)) f *= 2;
    }
    return f;
}

/// Total temporal downsample factor implied by the ladder (expect 4).
pub fn temporalFactor(cfg: Config) u32 {
    var f: u32 = 1;
    var i: u32 = 0;
    while (i < cfg.numBlocks()) : (i += 1) {
        if (temporalDownEffective(cfg, i)) f *= 2;
    }
    return f;
}

/// Geometry of one `Downsample3D`. Spatial padding is 0 by construction (the
/// encoder passes `downsample_padding=0`) and the asymmetric `(0,1,0,1)` pad is
/// applied separately — see `spatialPadAsymmetric`.
pub const DownsampleGeom = struct {
    kernel: [3]u32,
    stride: [3]u32,
    padding: [3]u32,
};

pub fn downsampleGeom(temporal_down: bool, spatial_down: bool, downsample_padding: u32) DownsampleGeom {
    return .{
        .kernel = .{
            if (temporal_down) 3 else 1,
            if (spatial_down) 3 else 1,
            if (spatial_down) 3 else 1,
        },
        .stride = .{
            if (temporal_down) 2 else 1,
            if (spatial_down) 2 else 1,
            if (spatial_down) 2 else 1,
        },
        .padding = .{
            if (temporal_down) 1 else 0,
            if (spatial_down) downsample_padding else 0,
            if (spatial_down) downsample_padding else 0,
        },
    };
}

/// diffusers' `Downsample2D` applies `(left, right, top, bottom) = (0,1,0,1)`
/// manually when its conv padding is 0. Symmetric padding here shifts the
/// feature map by half a pixel per block and compounds across the ladder.
pub const spatial_pad_asymmetric: [4]u32 = .{ 0, 1, 0, 1 };

/// `time_receptive_field`. THIS CHECKPOINT IS `.full`.
///
/// Every inner module (`ResnetBlock3D`, `DownEncoderBlock3D`, `UNetMidBlock3D`,
/// `Encoder3D`, `Decoder3D`) declares `= "half"` in its own signature, but
/// `VideoAutoencoderKL.__init__` declares `= "full"` and passes it down to the
/// encoder/decoder it builds, so the inner defaults are dead. The yaml sets
/// nothing. Reading `Encoder3D`'s signature and trusting its default — which is
/// what an earlier revision of this file did — gives `(1,3,3)` resnets and a
/// load failure.
pub const ReceptiveField = enum { half, full };

/// Resnet-block conv geometry. Under `.full` the resnets DO mix time.
/// Verified against ema_vae_fp16.safetensors: every
/// `encoder.down_blocks.*.resnets.*.conv{1,2}.weight` is `[C, C, 3, 3, 3]`.
pub fn resnetConvGeom(rf: ReceptiveField) DownsampleGeom {
    return switch (rf) {
        .half => .{ .kernel = .{ 1, 3, 3 }, .stride = .{ 1, 1, 1 }, .padding = .{ 0, 1, 1 } },
        .full => .{ .kernel = .{ 3, 3, 3 }, .stride = .{ 1, 1, 1 }, .padding = .{ 1, 1, 1 } },
    };
}

/// True when block `i` has a `conv_shortcut` — only where the channel count
/// changes. Blocks 0 (128->128) and 3 (512->512) keep their width and ship no
/// shortcut tensor; a loader that expects one per resnet finds nothing there.
pub fn hasConvShortcut(cfg: Config, i: u32) bool {
    if (i == 0) return false; // first block's input is block_out_channels[0]
    return cfg.block_out_channels[i - 1] != cfg.block_out_channels[i];
}

/// How many copies of frame 0 a causal conv prepends: `2 * temporal_padding`.
/// Causality in this VAE is head REPLICATION, not masking.
pub fn extendHeadTimes(temporal_padding: u32) u32 {
    return temporal_padding * 2;
}

/// Output temporal length of a causal conv, given the head extension. The
/// extension is sized exactly so a stride-1 conv is length-preserving; this
/// function exists to prove that rather than assume it.
pub fn causalOutLen(in_len: u32, kernel_t: u32, stride_t: u32, temporal_padding: u32) u32 {
    const extended = in_len + extendHeadTimes(temporal_padding);
    if (extended < kernel_t) return 0;
    return (extended - kernel_t) / stride_t + 1;
}

/// Pixel grid -> latent grid. Returns null when the input is not divisible by
/// the ladder's factors: this VAE has no internal padding, so an indivisible
/// input silently truncates rather than erroring.
pub fn latentShape(cfg: Config, t: u32, h: u32, w: u32) ?[3]u32 {
    const sf = spatialFactor(cfg);
    const tf = temporalFactor(cfg);
    if (h % sf != 0 or w % sf != 0) return null;
    // Temporal length is 4k+1 shaped: the causal head replication means the
    // first frame survives alone and the rest fold in groups of `tf`.
    if (t == 0) return null;
    if ((t - 1) % tf != 0) return null;
    return .{ (t - 1) / tf + 1, h / sf, w / sf };
}

/// Latent grid -> DiT token grid. `patch_size (1,2,2)` halves both spatial axes.
pub fn tokenGrid(latent: [3]u32) ?[3]u32 {
    if (latent[1] % 2 != 0 or latent[2] % 2 != 0) return null;
    return .{ latent[0], latent[1] / 2, latent[2] / 2 };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "the ladder yields /8 spatial and /4 temporal" {
    // The two headline factors from the yaml. If either of these moves, the
    // latent grid is wrong and every downstream token count with it.
    const cfg: Config = .{};
    try testing.expectEqual(@as(u32, 8), spatialFactor(cfg));
    try testing.expectEqual(@as(u32, 4), temporalFactor(cfg));
}

test "block 3 is flagged for temporal down but has no downsampler" {
    // THE TRAP. Both predicates must be ported; collapsing them into one
    // condition gives /8 in time and still produces a plausible video.
    const cfg: Config = .{};
    try testing.expect(temporalDownFlagged(cfg, 3));
    try testing.expect(!addDownsample(cfg, 3));
    try testing.expect(!temporalDownEffective(cfg, 3));
}

test "exactly blocks 1 and 2 stride in time; blocks 0..2 stride in space" {
    const cfg: Config = .{};
    const want_temporal = [_]bool{ false, true, true, false };
    const want_spatial = [_]bool{ true, true, true, false };
    for (want_temporal, want_spatial, 0..) |wt, ws, i| {
        const idx: u32 = @intCast(i);
        testing.expectEqual(wt, temporalDownEffective(cfg, idx)) catch |err| {
            std.debug.print("block {d}: temporal want {} got {}\n", .{ idx, wt, temporalDownEffective(cfg, idx) });
            return err;
        };
        testing.expectEqual(ws, addDownsample(cfg, idx)) catch |err| {
            std.debug.print("block {d}: spatial want {} got {}\n", .{ idx, ws, addDownsample(cfg, idx) });
            return err;
        };
    }
    // Block 0 is NOT flagged — the flag starts at index 1.
    try testing.expect(!temporalDownFlagged(cfg, 0));
    try testing.expect(temporalDownFlagged(cfg, 1));
}

test "oracle: this checkpoint's resnets are (3,3,3) — receptive field is full" {
    // ORACLE, from ema_vae_fp16.safetensors: every
    // encoder.down_blocks.*.resnets.*.conv1.weight is [C, C, 3, 3, 3].
    //
    // REGRESSION. This file previously asserted (1,3,3), taken from
    // Encoder3D's own `= "half"` default. VideoAutoencoderKL overrides it to
    // "full" for the modules it constructs, so the inner default is never the
    // one in force. Trust the checkpoint over the inner signature.
    const g = resnetConvGeom(.full);
    try testing.expectEqual([3]u32{ 3, 3, 3 }, g.kernel);
    try testing.expectEqual([3]u32{ 1, 1, 1 }, g.padding);
    // The "half" variant exists upstream but is NOT this checkpoint. Kept so
    // the distinction stays visible rather than being collapsed away.
    const half = resnetConvGeom(.half);
    try testing.expectEqual([3]u32{ 1, 3, 3 }, half.kernel);
    try testing.expectEqual([3]u32{ 0, 1, 1 }, half.padding);
}

test "oracle: downsampler temporal kernel tracks the effective ladder" {
    // ORACLE, same dump. Block 0's downsampler is [128,128,1,3,3] (spatial
    // only) while blocks 1 and 2 are [C,C,3,3,3]; block 3 has no downsampler
    // key at all. That is the ladder in §3.3 confirmed from the weights.
    const cfg: Config = .{};
    const want_temporal_kernel = [_]u32{ 1, 3, 3 };
    for (want_temporal_kernel, 0..) |wk, i| {
        const idx: u32 = @intCast(i);
        const g = downsampleGeom(temporalDownEffective(cfg, idx), true, 0);
        testing.expectEqual(wk, g.kernel[0]) catch |err| {
            std.debug.print("block {d}: temporal kernel want {d} got {d}\n", .{ idx, wk, g.kernel[0] });
            return err;
        };
    }
    try testing.expect(!addDownsample(cfg, 3));
}

test "oracle: conv_shortcut exists only where the channel count changes" {
    // ORACLE: the dump has conv_shortcut on blocks 1 (128->256) and 2
    // (256->512) only, each [C_out, C_in, 1, 1, 1].
    const cfg: Config = .{};
    try testing.expect(!hasConvShortcut(cfg, 0));
    try testing.expect(hasConvShortcut(cfg, 1));
    try testing.expect(hasConvShortcut(cfg, 2));
    try testing.expect(!hasConvShortcut(cfg, 3));
}

test "downsample geometry: temporal kernel/stride/pad appear together" {
    // A temporally-striding downsampler is k=3,s=2,p=1 in time; a purely
    // spatial one is k=1,s=1,p=0 — it must not convolve time at all.
    const t = downsampleGeom(true, true, 0);
    try testing.expectEqual([3]u32{ 3, 3, 3 }, t.kernel);
    try testing.expectEqual([3]u32{ 2, 2, 2 }, t.stride);
    try testing.expectEqual([3]u32{ 1, 0, 0 }, t.padding);

    const s = downsampleGeom(false, true, 0);
    try testing.expectEqual([3]u32{ 1, 3, 3 }, s.kernel);
    try testing.expectEqual([3]u32{ 1, 2, 2 }, s.stride);
    try testing.expectEqual([3]u32{ 0, 0, 0 }, s.padding);
}

test "encoder spatial padding is zero, carried by the asymmetric pad instead" {
    // downsample_padding=0 from Encoder3D. The (0,1,0,1) pad is applied in
    // forward; symmetric p=1 shifts the map half a pixel per block.
    const g = downsampleGeom(true, true, 0);
    try testing.expectEqual(@as(u32, 0), g.padding[1]);
    try testing.expectEqual(@as(u32, 0), g.padding[2]);
    try testing.expectEqual([4]u32{ 0, 1, 0, 1 }, spatial_pad_asymmetric);
}

test "causal head replication makes a stride-1 conv length preserving" {
    // extend_head prepends 2*temporal_padding copies of frame 0; for k=3,p=1
    // that is exactly the amount a zero-temporal-padding conv consumes.
    try testing.expectEqual(@as(u32, 2), extendHeadTimes(1));
    for ([_]u32{ 1, 2, 5, 17, 33 }) |t| {
        try testing.expectEqual(t, causalOutLen(t, 3, 1, 1));
    }
    // A conv that mixes no time (k=1, p=0) is trivially preserving and must
    // NOT extend the head.
    try testing.expectEqual(@as(u32, 0), extendHeadTimes(0));
    for ([_]u32{ 1, 5, 17 }) |t| {
        try testing.expectEqual(t, causalOutLen(t, 1, 1, 0));
    }
}

test "latent shape is 4k+1 in time and /8 in space" {
    const cfg: Config = .{};
    // A single frame stays a single latent frame — the image case.
    try testing.expectEqual([3]u32{ 1, 90, 160 }, latentShape(cfg, 1, 720, 1280).?);
    // 5 frames -> 2, 17 -> 5: (t-1)/4 + 1.
    try testing.expectEqual([3]u32{ 2, 90, 160 }, latentShape(cfg, 5, 720, 1280).?);
    try testing.expectEqual([3]u32{ 5, 90, 160 }, latentShape(cfg, 17, 720, 1280).?);
    // 1080p.
    try testing.expectEqual([3]u32{ 1, 135, 240 }, latentShape(cfg, 1, 1080, 1920).?);
}

test "oracle: latent shapes observed from the real VAE checkpoint" {
    // ORACLE. These are the shapes ByteDance's own VAE produced when run on
    // ema_vae_fp16.safetensors, dumped by
    //   tests/dump_seedvr2_fixtures.py vae
    // (see tests/fixtures/seedvr2/vae_meta.json). This is the independent
    // confirmation that the ladder above is not merely self-consistent with
    // my reading of the config — a /8-vs-/4 or 4k+1-vs-4k error would show up
    // here as a shape disagreement.
    //
    // NOTE the reference SQUEEZES the temporal axis when t == 1: it emits
    // [1,16,16,16] for a single frame and [1,16,2,8,8] for five. Our shape
    // function always reports the 3-axis grid, so the single-frame latent is
    // {1, h/8, w/8} and the squeeze is a presentation detail of the reference's
    // tensor layout, not a difference in the grid.
    const cfg: Config = .{};
    try testing.expectEqual([3]u32{ 1, 8, 8 }, latentShape(cfg, 1, 64, 64).?);
    try testing.expectEqual([3]u32{ 2, 8, 8 }, latentShape(cfg, 5, 64, 64).?);
    try testing.expectEqual([3]u32{ 1, 16, 16 }, latentShape(cfg, 1, 128, 128).?);
    // Non-square, and the case that would catch an h/w axis swap.
    try testing.expectEqual([3]u32{ 2, 16, 12 }, latentShape(cfg, 5, 128, 96).?);
}

test "latent shape refuses indivisible inputs instead of truncating" {
    // The VAE has no internal padding: an indivisible input silently loses
    // its trailing rows. Refuse by name at the boundary instead.
    const cfg: Config = .{};
    try testing.expect(latentShape(cfg, 1, 721, 1280) == null);
    try testing.expect(latentShape(cfg, 1, 720, 1281) == null);
    try testing.expect(latentShape(cfg, 4, 720, 1280) == null); // (4-1) % 4 != 0
    try testing.expect(latentShape(cfg, 0, 720, 1280) == null);
}

test "token grid halves the latent spatially and preserves time" {
    // patch_size (1,2,2). 720p -> 45x80, which is exactly the window module's
    // reference grid — the two files must agree on what 720p means.
    try testing.expectEqual([3]u32{ 1, 45, 80 }, tokenGrid(.{ 1, 90, 160 }).?);
    try testing.expectEqual([3]u32{ 5, 45, 80 }, tokenGrid(.{ 5, 90, 160 }).?);
    // 1080p latent 135x240 has an ODD height — it cannot be patchified, which
    // is why the pixel grid must be a multiple of 16, not 8.
    try testing.expect(tokenGrid(.{ 1, 135, 240 }) == null);
}

test "the 720p token grid agrees with the window module's reference constants" {
    // CROSS-FILE INVARIANT. seedvr2_window.zig derives its window size from a
    // hardcoded 45x80 "720p" grid. If this ladder ever stops producing 45x80
    // for 720p input, that constant is silently wrong and every window size
    // with it.
    const cfg: Config = .{};
    const latent = latentShape(cfg, 1, 720, 1280).?;
    const tokens = tokenGrid(latent).?;
    try testing.expectEqual([3]u32{ 1, 45, 80 }, tokens);
}

test "pixel grids must be multiples of 16 to survive VAE then patchify" {
    // Composition guard: /8 in the VAE then /2 in patchify. 8-divisible but
    // not 16-divisible inputs pass the VAE and die at the patch layer.
    const cfg: Config = .{};
    var h: u32 = 16;
    while (h <= 1088) : (h += 8) {
        const latent = latentShape(cfg, 1, h, 1280) orelse continue;
        const ok = tokenGrid(latent) != null;
        testing.expectEqual(h % 16 == 0, ok) catch |err| {
            std.debug.print("h={d}: divisible-by-16 {} but tokenGrid ok {}\n", .{ h, h % 16 == 0, ok });
            return err;
        };
    }
}
