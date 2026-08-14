//! LTX-2.5 DiffVAE decoder — geometry, config and schedule.
//!
//! `vae_diffusion_decoder.safetensors` (417M) is the decoder Lightricks' own
//! demos decode with; the conv `vae_decoder` we ship today is the cheap one.
//! Shape: four DETERMINISTIC neighborhood-attention stages upsample the latent
//! into a full-resolution context volume, then eight DIFFUSION blocks denoise
//! patchified noisy pixels against that context (AdaLN-Zero on a timestep
//! embedding, v-prediction, 2 Euler steps).
//!
//! THIS FILE IS GEOMETRY ONLY — no MLX, no forward pass. It owns the numbers
//! every part of the port has to agree on, because each of them is a silent
//! failure if wrong: a wrong window start reads the neighbouring pixels, a
//! wrong RoPE split rotates the wrong channels, a wrong stride ladder produces
//! a plausible video at the wrong scale. Reference:
//! `Lightricks/LTX-2` `ltx_core/model/video_vae/diffusion_video_decoder.py`
//! + `transformer/{layers,attention,rope_math}.py`, and
//! `transformer/fallback_na/eager.py` for the NATTEN window semantics.

const std = @import("std");

pub const Kernel = [3]u32; // (t, h, w)

pub const Stage = struct {
    dim: u32,
    depth: u32,
    kernel: Kernel,
};

/// Linear channel-expand then channels-last pixel shuffle.
pub const Upsample = struct {
    stride: [3]u32,
    reduction: u32,

    /// Output width of the stage's `proj` — the one thing about an upsample
    /// the checkpoint can confirm.
    pub fn projOut(self: Upsample, in_dim: u32) u32 {
        return in_dim * self.stride[0] * self.stride[1] * self.stride[2] / self.reduction;
    }
};

pub const Config = struct {
    stages: [4]Stage,
    upsamples: [4]Upsample,
    /// Context width the diffusion blocks cross-read (= last det stage's output).
    context_dim: u32,
    stage5_dim: u32,
    stage5_depth: u32,
    stage5_kernel: Kernel,
    head_dim: u32,
    patch_size: u32,
    t_emb_dim: u32,
    latent_channels: u32,
    out_channels: u32,
    rms_eps: f32 = 1e-6,
    rope_base: f32 = 10000.0,

    /// Channels of one patchified pixel token.
    pub fn patchChannels(self: Config) u32 {
        return self.out_channels * self.patch_size * self.patch_size;
    }

    /// Latent frames replicated before stage 1 to keep NATTEN's window from
    /// shifting inward at the LAST frame, cropped off the context again before
    /// the diffusion stage. `(stage1_K_t // 2) * 2`.
    pub fn trailingPadLatentFrames(self: Config) u32 {
        return (self.stages[0].kernel[0] / 2) * 2;
    }
};

/// The shipped LTX-2.5 decoder. Widths and depths are the CHECKPOINT's (read
/// off `vae_diffusion_decoder.safetensors`, pinned by the test below); the
/// upsample ladder and the kernels are the reference's production `L` layout.
///
/// The kernels are the one thing weights cannot confirm — no config ships with
/// the pack. They are taken as the reference defaults because everything that
/// CAN be checked agrees with that layout exactly: depths (4,6,4,2) and all
/// four (stride, reduction) pairs reproduce the four `upsamples.N.proj` shapes.
/// A checkpoint that disagreed would have to differ in a field that leaves no
/// trace in any tensor shape.
pub const production = Config{
    .stages = .{
        .{ .dim = 2048, .depth = 4, .kernel = .{ 3, 7, 7 } },
        .{ .dim = 1024, .depth = 6, .kernel = .{ 3, 7, 7 } },
        .{ .dim = 512, .depth = 4, .kernel = .{ 3, 5, 5 } },
        .{ .dim = 512, .depth = 2, .kernel = .{ 3, 5, 5 } },
    },
    .upsamples = .{
        .{ .stride = .{ 1, 2, 2 }, .reduction = 2 }, // space x2
        .{ .stride = .{ 2, 1, 1 }, .reduction = 2 }, // time x2
        .{ .stride = .{ 2, 2, 2 }, .reduction = 1 }, // all x2, width kept
        .{ .stride = .{ 2, 2, 2 }, .reduction = 2 }, // all x2
    },
    .context_dim = 256,
    .stage5_dim = 256,
    .stage5_depth = 8,
    .stage5_kernel = .{ 3, 7, 7 },
    .head_dim = 64,
    .patch_size = 4,
    .t_emb_dim = 384,
    .latent_channels = 128,
    .out_channels = 3,
};

/// First key index of the window query `i` attends, on one axis.
///
/// NATTEN does NOT clamp-and-mask: it SHIFTS the window inward so every query
/// sees exactly `kernel` keys. Porting the clamp-and-mask reading instead
/// gives a decoder that looks right everywhere except a `kernel/2`-wide frame
/// around each edge of every tile. Reference: `fallback_na/eager.py`
/// `_window_bounds` (non-causal branch).
pub fn naWindowStart(length: u32, kernel: u32, i: u32) u32 {
    const k = @min(kernel, length);
    const lo = length - k;
    const half = k / 2;
    const centered = if (i >= half) i - half else 0;
    return @min(centered, lo);
}

/// (T, H, W) split of `head_dim` across the three RoPE axes.
/// `d_t = (head_dim/4) rounded down to even`, the rest split evenly.
pub fn ropeDimSplit(head_dim: u32) [3]u32 {
    std.debug.assert(head_dim % 8 == 0);
    var d_t: u32 = (head_dim / 4) / 2 * 2;
    var d_hw: u32 = (head_dim - d_t) / 2;
    if (d_hw % 2 != 0) {
        d_t -= 2;
        d_hw = (head_dim - d_t) / 2;
    }
    return .{ d_t, d_hw, d_hw };
}

/// `1 / base^(2j/dim)` — the j-th inverse frequency of a `dim`-wide axis chunk.
pub fn ropeInvFreq(dim: u32, base: f64, j: u32) f64 {
    const e = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dim));
    return 1.0 / std.math.pow(f64, base, e);
}

/// Volume after one pixel-shuffle upsample. A `stride[0] == 2` shuffle emits a
/// DUPLICATE leading frame that only the chunk holding t=0 may drop — that is
/// what keeps the composed 1:8 latent→pixel frame mapping causal.
pub fn upsampleOut(dims: [3]u32, up: Upsample, drop_leading_frame: bool) [3]u32 {
    var out = [3]u32{
        dims[0] * up.stride[0],
        dims[1] * up.stride[1],
        dims[2] * up.stride[2],
    };
    if (up.stride[0] == 2 and drop_leading_frame) out[0] -= 1;
    return out;
}

/// Token volume entering each stage, then the diffusion stage — index 0..3 are
/// the det stages' inputs, index 4 is the context (= diffusion) volume.
pub fn stageVolumes(cfg: Config, latent: [3]u32, drop_leading_frame: bool) [5][3]u32 {
    var out: [5][3]u32 = undefined;
    out[0] = latent;
    for (0..4) |i| out[i + 1] = upsampleOut(out[i], cfg.upsamples[i], drop_leading_frame);
    return out;
}

/// Pixel volume (frames, height, width) a latent decodes to.
pub fn pixelShape(cfg: Config, latent: [3]u32) [3]u32 {
    const v = stageVolumes(cfg, latent, true)[4];
    return .{ v[0], v[1] * cfg.patch_size, v[2] * cfg.patch_size };
}

/// Whether every stage's volume clears its own kernel on every axis — NA has
/// no answer for a volume smaller than its window, and the reference raises.
/// A clip that fails this is edge-padded before decode, not refused.
pub fn volumesClearKernels(cfg: Config, latent: [3]u32) bool {
    const v = stageVolumes(cfg, latent, true);
    for (0..4) |i| {
        for (0..3) |a| if (v[i][a] < cfg.stages[i].kernel[a]) return false;
    }
    for (0..3) |a| if (v[4][a] < cfg.stage5_kernel[a]) return false;
    return true;
}

/// `linspace(1.0, 1/n, n)` — the timestep the model is asked to denoise FROM at
/// each step. The step after the last one targets 0. How MANY steps, what the
/// model predicts and what the timestep is scaled by are NOT geometry: no pack
/// ships a VAE config, so they are measured, and `ltx_diffvae_forward.Sampler`
/// owns them.
pub fn timesteps(n: u32, out: []f32) []f32 {
    std.debug.assert(out.len >= n);
    const last = 1.0 / @as(f32, @floatFromInt(n));
    if (n == 1) {
        out[0] = 1.0;
        return out[0..1];
    }
    const span = 1.0 - last;
    for (0..n) |i| {
        out[i] = 1.0 - span * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
    }
    return out[0..n];
}

/// One reverse Euler update on a VELOCITY prediction: `x - (t_now - t_next)*v`.
pub fn eulerStepScale(t_now: f32, t_next: f32) f32 {
    return t_now - t_next;
}

// ── Tiling geometry ──────────────────────────────────────────────────────
//
// Only stage 4 and the diffusion stage are tiled: stages 1-3 run on the FULL
// volume (1.6M tokens at dim 512 for a 1920x1088 clip — a couple of GB), while
// the diffusion stage is 12.7M tokens at dim 256 through 8 blocks twice, which
// is the only thing that has to be cut. Tiles are cut on the STAGE-4 INPUT grid
// and land on pixel coordinates through one upsample hop plus the unpatchify.

/// Per-axis latent-grid floor so every stage's NA sees dims >= its own kernel.
/// A latent under this is edge-padded before decode, and the pad cropped off
/// the pixels at the end.
pub fn allStagesMinTile(cfg: Config) [3]u32 {
    var mins = [3]u32{ 1, 1, 1 };
    var cum = [3]u32{ 1, 1, 1 };
    for (0..4) |i| {
        for (0..3) |a| mins[a] = @max(mins[a], divCeil(cfg.stages[i].kernel[a], cum[a]));
        for (0..3) |a| cum[a] *= cfg.upsamples[i].stride[a];
    }
    for (0..3) |a| mins[a] = @max(mins[a], divCeil(cfg.stage5_kernel[a], cum[a]));
    return mins;
}

/// Min stage-4-INPUT extent so stage 4 and the diffusion stage each still clear
/// their kernel — the floor a tile may not be split below.
pub fn tileMinSize(cfg: Config) [3]u32 {
    var out: [3]u32 = undefined;
    const up = cfg.upsamples[3].stride;
    for (0..3) |a| out[a] = @max(cfg.stages[3].kernel[a], divCeil(cfg.stage5_kernel[a], up[a]));
    return out;
}

/// One-sided halo, in stage-4-input units, that stage 4 and the diffusion stage
/// each reach across: `depth * (kernel/2)` hops. The tile overlap is built from
/// the larger of the two.
pub fn tileHalos(cfg: Config) [2][3]u32 {
    var out: [2][3]u32 = undefined;
    const up = cfg.upsamples[3].stride;
    for (0..3) |a| {
        out[0][a] = cfg.stages[3].depth * (cfg.stages[3].kernel[a] / 2);
        out[1][a] = divCeil(cfg.stage5_depth * (cfg.stage5_kernel[a] / 2), up[a]);
    }
    return out;
}

/// Stage-4 input `(T, H, W)` — the latent after the first three upsample hops.
pub fn stage4FromLatent(cfg: Config, latent: [3]u32, drop_leading_frame: bool) [3]u32 {
    var out = latent;
    for (0..3) |i| out = upsampleOut(out, cfg.upsamples[i], drop_leading_frame);
    return out;
}

/// Pixel/frame units per stage-4-input cell: the last NA hop then unpatchify.
pub fn stage4PixelScale(cfg: Config) [3]u32 {
    const up = cfg.upsamples[3].stride;
    return .{ up[0], up[1] * cfg.patch_size, up[2] * cfg.patch_size };
}

/// Composed latent→context temporal scale — the product of the four upsample
/// temporal strides (the dropped leading frame is not part of the RATIO).
pub fn latentTimeScale(cfg: Config) u32 {
    var t: u32 = 1;
    for (cfg.upsamples) |u| t *= u.stride[0];
    return t;
}

/// Composed latent→pixel scale on (h, w): the upsample ladder times the patch.
pub fn latentSpatialScale(cfg: Config) [2]u32 {
    var out = [2]u32{ cfg.patch_size, cfg.patch_size };
    for (cfg.upsamples) |u| {
        out[0] *= u.stride[1];
        out[1] *= u.stride[2];
    }
    return out;
}

/// Frames of the ghost appendix (the replicated trailing latent frames) to crop
/// off a context volume, never taking it below the diffusion stage's own kernel.
pub fn contextKeepFrames(cfg: Config, ctx_frames: u32, time_scale: u32) u32 {
    const pad = cfg.trailingPadLatentFrames();
    if (pad == 0) return ctx_frames;
    const ghost = pad * time_scale;
    const content = if (ctx_frames > ghost) ctx_frames - ghost else 1;
    return @min(ctx_frames, @max(content, cfg.stage5_kernel[0]));
}

/// One tile's extent on one axis: content on the stage-4 grid, where it lands in
/// pixel space, and the blend ramps (pixel units) at each end.
pub const Interval = struct {
    start: u32,
    end: u32,
    out_start: u32,
    out_end: u32,
    left_ramp: u32 = 0,
    right_ramp: u32 = 0,

    pub fn len(self: Interval) u32 {
        return self.end - self.start;
    }
    pub fn outLen(self: Interval) u32 {
        return self.out_end - self.out_start;
    }
};

/// How a stage-4 coordinate maps into pixel space on one axis. Temporal is the
/// pixel-shuffle mapping: the whole axis loses ONE frame, the duplicate leading
/// one that only the chunk owning t=0 drops — so every later coordinate is
/// shifted down by 1 rather than each tile dropping a frame of its own.
pub const AxisMap = struct {
    scale: u32,
    temporal: bool = false,

    pub fn map(self: AxisMap, c: u32) u32 {
        if (self.temporal and self.scale == 2) return if (c == 0) 0 else c * 2 - 1;
        return c * self.scale;
    }
};

fn divCeil(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// Split one stage-4 axis into equal `max_tile`-long tiles whose seams overlap
/// by at least `overlap`, with matching trapezoid ramps on both sides of every
/// seam so the tile weights sum to EXACTLY 1 — no weight buffer, no division.
/// Returns the prefix of `out` that was filled.
pub fn splitAxis(
    length: u32,
    max_tile_in: u32,
    overlap: u32,
    min_size: u32,
    m: AxisMap,
    out: []Interval,
) []Interval {
    std.debug.assert(out.len >= 1);
    // A tile shorter than twice the overlap would carry ramps that meet in the
    // middle, and two meeting ramps are not a partition of unity.
    const max_tile = @max(@max(max_tile_in, 2 * overlap), min_size);
    if (length <= max_tile or overlap == 0 or out.len == 1) {
        out[0] = .{ .start = 0, .end = length, .out_start = 0, .out_end = m.map(length) };
        return out[0..1];
    }
    const step = max_tile - overlap;
    var n = divCeil(length - overlap, step);
    if (n > out.len) n = @intCast(out.len);
    if (n < 2) {
        out[0] = .{ .start = 0, .end = length, .out_start = 0, .out_end = m.map(length) };
        return out[0..1];
    }
    // Even starts across the slack, so every tile is exactly `max_tile` long and
    // the seams share one overlap width each.
    const slack = length - max_tile;
    for (0..n) |i| {
        const start: u32 = @intCast(slack * i / (n - 1));
        out[i] = .{
            .start = start,
            .end = start + max_tile,
            .out_start = m.map(start),
            .out_end = m.map(start + max_tile),
        };
    }
    for (0..n - 1) |i| {
        const seam = out[i].out_end - out[i + 1].out_start;
        out[i].right_ramp = seam;
        out[i + 1].left_ramp = seam;
    }
    return out[0..n];
}

/// Blend weight of pixel `i` (tile-local) under the interval's trapezoid.
/// Half-offset ramps: the two sides of a seam are `(j+0.5)/L` and
/// `(L-j-0.5)/L`, which sum to 1 for every j — that is the whole trick.
pub fn tileWeight(iv: Interval, i: u32) f32 {
    var w: f32 = 1.0;
    if (iv.left_ramp > 0 and i < iv.left_ramp) {
        w = (@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(iv.left_ramp));
    }
    const n = iv.outLen();
    if (iv.right_ramp > 0 and i + iv.right_ramp >= n) {
        const j = i + iv.right_ramp - n;
        const r = (@as(f32, @floatFromInt(iv.right_ramp - j)) - 0.5) / @as(f32, @floatFromInt(iv.right_ramp));
        w = @min(w, r);
    }
    return w;
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "diffvae config reproduces the checkpoint's upsample projection shapes" {
    // Recorded from vae_diffusion_decoder.safetensors (mlx-community/ltx-2.5-mlx):
    // upsamples.{0..3}.proj.weight = [out, in].
    const recorded = [4][2]u32{
        .{ 4096, 2048 },
        .{ 1024, 1024 },
        .{ 4096, 512 },
        .{ 2048, 512 },
    };
    for (0..4) |i| {
        try testing.expectEqual(recorded[i][1], production.stages[i].dim);
        try testing.expectEqual(recorded[i][0], production.upsamples[i].projOut(production.stages[i].dim));
    }
    // conv_in / conv_in_x_t / conv_out widths.
    try testing.expectEqual(@as(u32, 48), production.patchChannels());
    try testing.expectEqual(@as(u32, 2048), production.stages[0].dim);
    // Every stage width must divide into whole heads.
    for (production.stages) |s| try testing.expectEqual(@as(u32, 0), s.dim % production.head_dim);
    try testing.expectEqual(@as(u32, 0), production.stage5_dim % production.head_dim);
}

test "na window shifts inward at the edges instead of shrinking" {
    // Reference `_window_bounds(length=8, kernel=5)`: starts 0,0,0,1,2,3,3,3.
    const expect = [_]u32{ 0, 0, 0, 1, 2, 3, 3, 3 };
    for (expect, 0..) |want, i| {
        try testing.expectEqual(want, naWindowStart(8, 5, @intCast(i)));
    }
    // Every query sees exactly `kernel` keys, all in range — that is the whole
    // point of the shift, and the property a clamp-and-mask port would break.
    for (0..8) |i| {
        const s = naWindowStart(8, 5, @intCast(i));
        try testing.expect(s + 5 <= 8);
    }
    // A volume shorter than the kernel degenerates to "attend everything".
    try testing.expectEqual(@as(u32, 0), naWindowStart(3, 7, 2));
}

test "rope split covers head_dim with even per-axis chunks" {
    const s = ropeDimSplit(64);
    try testing.expectEqual([3]u32{ 16, 24, 24 }, s);
    try testing.expectEqual(@as(u32, 64), s[0] + s[1] + s[2]);
    for (s) |d| try testing.expectEqual(@as(u32, 0), d % 2);
    // First frequency is 1.0 on every axis; they decay from there.
    try testing.expectApproxEqAbs(@as(f64, 1.0), ropeInvFreq(16, 10000.0, 0), 1e-12);
    try testing.expect(ropeInvFreq(16, 10000.0, 7) < ropeInvFreq(16, 10000.0, 1));
}

test "the stage ladder lands on the conv VAE's own 8x32 compression" {
    // 97 frames at 768x512 → latent (13, 16, 24) out of the conv encoder.
    const px = pixelShape(production, .{ 13, 16, 24 });
    try testing.expectEqual([3]u32{ 97, 512, 768 }, px);
    // Spatial: 32x per axis. Temporal: three time-doubling shuffles, each
    // dropping its duplicate leading frame — 13 → 25 → 49 → 97, which lands
    // exactly on the 8N+1 clip length the request asked for.
    try testing.expectEqual(@as(u32, 16 * 32), px[1]);
    try testing.expectEqual(@as(u32, 24 * 32), px[2]);

    // Intermediate volumes, which is what the memory bill is made of.
    const v = stageVolumes(production, .{ 13, 16, 24 }, true);
    try testing.expectEqual([3]u32{ 13, 16, 24 }, v[0]);
    try testing.expectEqual([3]u32{ 13, 32, 48 }, v[1]);
    try testing.expectEqual([3]u32{ 25, 32, 48 }, v[2]);
    try testing.expectEqual([3]u32{ 49, 64, 96 }, v[3]);
    try testing.expectEqual([3]u32{ 97, 128, 192 }, v[4]);
}

test "a volume smaller than a stage kernel is caught before NA sees it" {
    try testing.expect(volumesClearKernels(production, .{ 13, 16, 24 }));
    // One latent frame: stage 1 has T=1 against K_t=3, and every later stage
    // stays under it too — this is the case the reference edge-pads for.
    try testing.expect(!volumesClearKernels(production, .{ 1, 16, 24 }));
    // Thin canvas: W=2 at stage 1 is under K_w=7.
    try testing.expect(!volumesClearKernels(production, .{ 13, 16, 2 }));
}

test "timesteps are the reference linspace and the euler step is their gap" {
    var buf: [8]f32 = undefined;
    const t = timesteps(2, &buf);
    try testing.expectEqual(@as(usize, 2), t.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), t[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), t[1], 1e-6);
    // Step 1 runs 1.0 → 0.5, step 2 runs 0.5 → 0 (the schedule's terminal).
    try testing.expectApproxEqAbs(@as(f32, 0.5), eulerStepScale(t[0], t[1]), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), eulerStepScale(t[1], 0.0), 1e-6);
    const one = timesteps(1, &buf);
    try testing.expectApproxEqAbs(@as(f32, 1.0), one[0], 1e-6);
}

test "the trailing natten pad is the stage-1 window's own half-width" {
    try testing.expectEqual(@as(u32, 2), production.trailingPadLatentFrames());
}

test "the tiling floors and halos come from the stage ladder, not a constant" {
    // Latent floor: every stage's kernel divided back through the upsamples it
    // sits behind. Stage 1 sees the latent itself, so K=(3,7,7) is the binding
    // constraint on H/W; the temporal axis relaxes as the time-doubling hops
    // accumulate.
    try testing.expectEqual([3]u32{ 3, 7, 7 }, allStagesMinTile(production));
    // Stage-4 input floor: stage 4's own (3,5,5) against the diffusion stage's
    // (3,7,7) divided by the last hop's 2x.
    try testing.expectEqual([3]u32{ 3, 5, 5 }, tileMinSize(production));
    // Halos: stage 4 = depth 2 x (kernel/2); diffusion = depth 8 x (kernel/2)
    // brought back through the 2x hop.
    const halos = tileHalos(production);
    try testing.expectEqual([3]u32{ 2, 4, 4 }, halos[0]);
    try testing.expectEqual([3]u32{ 4, 12, 12 }, halos[1]);
    // Stage-4 grid and its pixel scale reproduce the full ladder.
    try testing.expectEqual([3]u32{ 49, 64, 96 }, stage4FromLatent(production, .{ 13, 16, 24 }, true));
    try testing.expectEqual([3]u32{ 2, 8, 8 }, stage4PixelScale(production));
    const s4 = stage4FromLatent(production, .{ 13, 16, 24 }, true);
    const sc = stage4PixelScale(production);
    try testing.expectEqual(pixelShape(production, .{ 13, 16, 24 }), [3]u32{ s4[0] * sc[0] - 1, s4[1] * sc[1], s4[2] * sc[2] });
}

test "the ghost-frame crop never takes the context below the diffusion kernel" {
    // 2 replicated latent frames x the 8x temporal scale = 16 ghost frames.
    try testing.expectEqual(@as(u32, 97), contextKeepFrames(production, 113, 8));
    // A clip so short the appendix is most of it keeps stage5_kernel[0] frames
    // rather than starving NA.
    try testing.expectEqual(@as(u32, 3), contextKeepFrames(production, 17, 8));
    // Nothing to crop when the context is already shorter than the kernel.
    try testing.expectEqual(@as(u32, 2), contextKeepFrames(production, 2, 8));
}

test "tile weights are a partition of unity over the pixel axis" {
    // A seam blend that does not sum to 1 is a visible band down the middle of
    // the frame, and a cosine over the whole picture will not see it.
    const cases = [_]struct { len: u32, tile: u32, ov: u32, m: AxisMap }{
        .{ .len = 96, .tile = 24, .ov = 8, .m = .{ .scale = 8 } },
        .{ .len = 96, .tile = 40, .ov = 12, .m = .{ .scale = 8 } },
        .{ .len = 49, .tile = 16, .ov = 4, .m = .{ .scale = 2, .temporal = true } },
        .{ .len = 33, .tile = 32, .ov = 4, .m = .{ .scale = 2, .temporal = true } },
        .{ .len = 7, .tile = 24, .ov = 8, .m = .{ .scale = 8 } }, // untiled
    };
    var buf: [32]Interval = undefined;
    for (cases) |c| {
        const tiles = splitAxis(c.len, c.tile, c.ov, 3, c.m, &buf);
        const total = c.m.map(c.len);
        const acc = try testing.allocator.alloc(f32, total);
        defer testing.allocator.free(acc);
        @memset(acc, 0);
        for (tiles) |iv| {
            try testing.expect(iv.left_ramp + iv.right_ramp <= iv.outLen());
            for (0..iv.outLen()) |i| acc[iv.out_start + i] += tileWeight(iv, @intCast(i));
        }
        // Every pixel is covered exactly once in weight, and the tiles cover
        // the whole axis with no hole between them.
        for (acc, 0..) |w, i| {
            testing.expectApproxEqAbs(@as(f32, 1.0), w, 1e-6) catch |e| {
                std.debug.print("[diffvae-tile] len={d} tile={d} pixel {d} weight {d}\n", .{ c.len, c.tile, i, w });
                return e;
            };
        }
        try testing.expectEqual(@as(u32, 0), tiles[0].out_start);
        try testing.expectEqual(total, tiles[tiles.len - 1].out_end);
    }
}

test "a temporal tile map keeps the single dropped frame global, not per tile" {
    // The pixel-shuffle emits ONE duplicate leading frame for the whole volume;
    // only the chunk owning t=0 drops it. A per-tile drop would shorten the clip
    // by one frame per seam.
    const m = AxisMap{ .scale = 2, .temporal = true };
    try testing.expectEqual(@as(u32, 0), m.map(0));
    try testing.expectEqual(@as(u32, 1), m.map(1));
    try testing.expectEqual(@as(u32, 97), m.map(49));
    var buf: [8]Interval = undefined;
    const tiles = splitAxis(49, 16, 4, 3, m, &buf);
    try testing.expect(tiles.len > 1);
    // Contiguous, gap-free, and the union is exactly the untiled frame count.
    for (tiles[1..], 0..) |iv, i| try testing.expect(iv.out_start < tiles[i].out_end);
    try testing.expectEqual(@as(u32, 97), tiles[tiles.len - 1].out_end);
}

test "the composed latent scales are the ladder's own product" {
    // 1x2x2x2 temporal, (2*1*2*2)*4 = 32 spatial — the conv VAE's own 8x32.
    try testing.expectEqual(@as(u32, 8), latentTimeScale(production));
    try testing.expectEqual([2]u32{ 32, 32 }, latentSpatialScale(production));
}
