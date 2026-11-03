//! RADIO tower + nemotron_omni projector for Nemotron 3.5 Bolt and Super.
//!
//! Weights keep the checkpoint names (`vision_model.radio_model.model.*`,
//! `mlp1.*`). The transformers rename in modeling_radio.py is not applied.
//! A CHW float32 buffer already normalized with CLIP mean/std becomes
//! `[1, tokens, llm_hidden]` after a 32-block ViT, v2 pixel-shuffle, and the
//! ReLU² projector. Class and register tokens are dropped. Video packs
//! `video_temporal_patch_size` frames on the channel axis and projects them
//! with `patch_generator.video_embedder` instead of the image embedder.
//! The same tower and projector then run once per tubelet.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const Weights = model_mod.Weights;
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const glm_vision = @import("../glm5_next/vision.zig");
const transformer = @import("../../engines/mlx/transformer.zig");
const activations = @import("../../sdk/blocks/mlp/activations.zig");
const params = @import("../../sdk/quant/params.zig");
const log = @import("../../sdk/util/log.zig");

const ModelConfig = model_mod.ModelConfig;

/// Patch budget. The processor allows 13312, which is a full bidirectional
/// attention over the language model already resident. 4096 patches is 1024
/// language tokens after the 2×2 shuffle.
pub const ENGINE_MAX_PATCHES: u32 = 4096;
pub const MIN_PATCHES: u32 = 1024;
pub const PATCH: u32 = 16;
pub const DOWNSAMPLE: u32 = 2;
/// Frames folded into one tubelet. Matches `video_temporal_patch_size`.
pub const TEMPORAL: u32 = 2;
/// Patch budget for a video frame (`video_target_num_patches`), pre-shuffle.
pub const VIDEO_TARGET_PATCHES: u32 = 1024;
const PROJECTOR_EPS: f32 = 1e-5;

/// OPENAI / RADIO CLIP normalization. The tower's input_conditioner is unused;
/// the processor applies this itself.
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

pub const PatchGrid = struct { w: u32, h: u32 };

/// Python 3 `round` (half to even).
fn roundHalfEven(x: f64) i32 {
    if (!std.math.isFinite(x)) return 0;
    const down = @floor(x);
    const frac = x - down;
    const base: i32 = @intFromFloat(down);
    if (frac < 0.5) return base;
    if (frac > 0.5) return base + 1;
    if (@mod(base, 2) == 0) return base;
    return base + 1;
}

fn snapToDivisor(dim: *u32, other: u32, budget: u32) void {
    const rem = dim.* % DOWNSAMPLE;
    if (rem == 0) return;
    const inc = DOWNSAMPLE - rem;
    if ((dim.* + inc) * other <= budget) {
        dim.* += inc;
    } else {
        dim.* = @max(DOWNSAMPLE, dim.* - rem);
    }
}

/// `(w_patches, h_patches)` from `NemotronH_Omni_Reasoning_V3ImageProcessor._compute_target_patches`.
/// `budget` is a patch count (pre-shuffle), clamped to `[MIN_PATCHES, ENGINE_MAX_PATCHES]`.
pub fn targetPatches(src_h: u32, src_w: u32, budget_in: u32) PatchGrid {
    const budget = std.math.clamp(if (budget_in == 0) ENGINE_MAX_PATCHES else budget_in, MIN_PATCHES, ENGINE_MAX_PATCHES);
    if (src_h == 0 or src_w == 0) return .{ .w = 32, .h = 32 };
    const closest_h: u32 = @intCast(@max(roundHalfEven(@as(f64, @floatFromInt(src_h)) / PATCH + 0.5), 1));
    const closest_w: u32 = @intCast(@max(roundHalfEven(@as(f64, @floatFromInt(src_w)) / PATCH + 0.5), 1));
    const patches: f64 = @floatFromInt(closest_h * closest_w);
    const factor = @min(@sqrt(@as(f64, @floatFromInt(budget)) / patches), 1.0);
    var th: u32 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_h))));
    var tw: u32 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_w))));
    if (th == 0) th = 1;
    if (tw == 0) tw = 1;
    if (budget > MIN_PATCHES and th * tw < MIN_PATCHES) {
        const up = @sqrt(@as(f64, @floatFromInt(MIN_PATCHES)) / @as(f64, @floatFromInt(th * tw)));
        th = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(th))));
        tw = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(tw))));
    }
    snapToDivisor(&th, tw, budget);
    snapToDivisor(&tw, th, budget);
    if (th == 0 or tw == 0) return .{ .w = 32, .h = 32 };
    return .{ .w = tw, .h = th };
}

fn roundAtLeast1(x: f64) u32 {
    const r = roundHalfEven(x);
    if (r < 1) return 1;
    return @intCast(r);
}

/// `(w_patches, h_patches)` from `_compute_target_patches_video`.
/// Aspect is preserved and both dims snap to `DOWNSAMPLE`. The budget is a
/// fixed patch count, not a cap against the source size: a small frame is
/// scaled up to the same grid as a large one of the same aspect.
pub fn targetPatchesVideo(src_h: u32, src_w: u32) PatchGrid {
    const target: u32 = VIDEO_TARGET_PATCHES;
    const divisor: u32 = DOWNSAMPLE;
    if (src_h == 0 or src_w == 0) return .{ .w = 32, .h = 32 };
    const aspect = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(@max(src_h, 1)));
    var ph: u32 = roundAtLeast1(@sqrt(@as(f64, @floatFromInt(target)) / aspect));
    var pw: u32 = roundAtLeast1(@sqrt(@as(f64, @floatFromInt(target)) * aspect));
    if (divisor > 1) {
        const rem_h = ph % divisor;
        const rem_w = pw % divisor;
        const ph_up = ph + if (rem_h != 0) divisor - rem_h else 0;
        const ph_down = ph - rem_h;
        const pw_up = pw + if (rem_w != 0) divisor - rem_w else 0;
        const pw_down = pw - rem_w;
        if (ph_up * pw_up <= target) {
            ph = ph_up;
            pw = pw_up;
        } else {
            ph = @max(divisor, ph_down);
            pw = @max(divisor, pw_down);
        }
    }
    // Past `target / divisor` aspect the short side sits on its floor; trim the long one.
    if (ph * pw > target) {
        if (pw > ph) pw = @max(divisor, target / ph / divisor * divisor) else ph = @max(divisor, target / pw / divisor * divisor);
    }
    if (ph == 0 or pw == 0) return .{ .w = 32, .h = 32 };
    return .{ .w = pw, .h = ph };
}

/// Tubelet grid for pixels already resized to a video patch target.
/// `pixel_h` / `pixel_w` are the shared frame size. `spatial` is language
/// tokens per tubelet after the 2×2 shuffle.
pub const VideoGeom = struct {
    patch_w: u32,
    patch_h: u32,
    token_w: u32,
    token_h: u32,
    spatial: u32,
    groups: u32,
    frames: u32,
};

pub fn videoGeometry(frames: u32, pixel_h: u32, pixel_w: u32) ?VideoGeom {
    if (frames == 0 or pixel_h == 0 or pixel_w == 0) return null;
    if (pixel_h % PATCH != 0 or pixel_w % PATCH != 0) return null;
    const patch_h = pixel_h / PATCH;
    const patch_w = pixel_w / PATCH;
    if (patch_h % DOWNSAMPLE != 0 or patch_w % DOWNSAMPLE != 0) return null;
    const token_h = patch_h / DOWNSAMPLE;
    const token_w = patch_w / DOWNSAMPLE;
    if (token_h == 0 or token_w == 0) return null;
    return .{
        .patch_w = patch_w,
        .patch_h = patch_h,
        .token_w = token_w,
        .token_h = token_h,
        .spatial = token_h * token_w,
        .groups = (frames + TEMPORAL - 1) / TEMPORAL,
        .frames = frames,
    };
}

/// One tubelet's label, exactly the processor's no-timestamp form.
/// Group 0 is `"Frame 1 and frame 2: "`. Later groups lead with `"\n"`.
/// A short final group omits the padded frame: three frames end at `"\nFrame 3: "`.
/// Returns empty when `buf` cannot hold the text.
pub fn videoGroupSegment(buf: []u8, group: u32, frames: u32) []const u8 {
    var n: usize = 0;
    if (group > 0) {
        if (buf.len < 1) return buf[0..0];
        buf[0] = '\n';
        n = 1;
    }
    var j: u32 = 0;
    var any = false;
    while (j < TEMPORAL) : (j += 1) {
        const fi = group * TEMPORAL + j;
        if (fi >= frames) break;
        if (any) {
            if (n + 5 > buf.len) return buf[0..0];
            @memcpy(buf[n..][0..5], " and ");
            n += 5;
        }
        const word: []const u8 = if (j == 0) "Frame " else "frame ";
        if (n + word.len > buf.len) return buf[0..0];
        @memcpy(buf[n..][0..word.len], word);
        n += word.len;
        const num = std.fmt.bufPrint(buf[n..], "{d}", .{fi + 1}) catch return buf[0..0];
        n += num.len;
        any = true;
    }
    if (!any or n + 2 > buf.len) return buf[0..0];
    @memcpy(buf[n..][0..2], ": ");
    n += 2;
    return buf[0..n];
}

/// How many language tokens EVS keeps. `int(groups * spatial * (1-q))`, and
/// never fewer than one tubelet. `q <= 0` keeps the whole clip.
pub fn evsKeepCount(groups: u32, spatial: u32, q: f64) u32 {
    const all = groups * spatial;
    if (!(q > 0) or groups == 0 or spatial == 0) return all;
    if (q >= 1) return spatial;
    const prod = @as(f64, @floatFromInt(groups)) * @as(f64, @floatFromInt(spatial)) * (1.0 - q);
    const evs: u32 = @intFromFloat(@trunc(prod));
    return @max(spatial, evs);
}

/// Stable descending top-`keep` over `scores`. True means the token stays,
/// in the original order (the official mask is not a reorder).
pub fn evsRetentionMask(allocator: std.mem.Allocator, scores: []const f32, keep: usize) ![]bool {
    const mask = try allocator.alloc(bool, scores.len);
    if (keep >= scores.len) {
        @memset(mask, true);
        return mask;
    }
    const order = try allocator.alloc(usize, scores.len);
    defer allocator.free(order);
    for (order, 0..) |*slot, i| slot.* = i;
    std.mem.sort(usize, order, scores, struct {
        fn less(s: []const f32, a: usize, b: usize) bool {
            if (s[a] == s[b]) return a < b;
            return s[a] > s[b];
        }
    }.less);
    @memset(mask, false);
    for (order[0..keep]) |i| mask[i] = true;
    return mask;
}

fn cosine(a: []const f32, b: []const f32) f32 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        const xf: f64 = x;
        const yf: f64 = y;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    const denom = @max(@sqrt(na), 1e-8) * @max(@sqrt(nb), 1e-8);
    return @floatCast(dot / denom);
}

/// Dissimilarity scores for one clip, length `groups * spatial`.
/// `embeds` is row-major `[groups, spatial, hidden]`. The first tubelet is
/// 255 so it always outranks a later frame (cosine dissimilarity is at most 2).
pub fn videoDissimilarity(embeds: []const f32, groups: usize, spatial: usize, hidden: usize, scores: []f32) void {
    for (0..spatial) |i| scores[i] = 255;
    var t: usize = 1;
    while (t < groups) : (t += 1) {
        for (0..spatial) |i| {
            const prev = embeds[((t - 1) * spatial + i) * hidden ..][0..hidden];
            const cur = embeds[(t * spatial + i) * hidden ..][0..hidden];
            const d = 1 - cosine(prev, cur);
            scores[t * spatial + i] = if (std.math.isFinite(d)) d else 0;
        }
    }
}

pub const Prepared = struct {
    pixels: []u8,
    h: u32,
    w: u32,
    tokens: u32,
};

fn resizeClipChw(a: std.mem.Allocator, chw: []f32, rgb: []const u8, sh: u32, sw: u32, th: u32, tw: u32) !void {
    try qwen_vision.resizeRgbBicubicFloatChw(a, chw, rgb, sh, sw, th, tw);
    const plane: usize = @as(usize, th) * tw;
    for (chw, 0..) |*value, i| value.* = (value.* / 255 - MEAN[i / plane]) / STD[i / plane];
}

/// Antialiased float32 bicubic resize, then CLIP normalization.
/// Returned bytes are float32 CHW `[3, H, W]`. `tokens` is the language-token count after shuffle.
pub fn prepareImage(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32) !Prepared {
    const grid = targetPatches(src_h, src_w, ENGINE_MAX_PATCHES);
    const tw = grid.w * PATCH;
    const th = grid.h * PATCH;
    const plane: usize = @as(usize, th) * tw;
    // Keep the allocation align-1 so callers can free the []u8 pixels. The
    // write needs an f32 view, and a byte slice does not carry that alignment.
    const buf = try allocator.alloc(u8, plane * 3 * @sizeOf(f32));
    errdefer allocator.free(buf);
    const aligned: []align(@alignOf(f32)) u8 = @alignCast(buf);
    try resizeClipChw(allocator, std.mem.bytesAsSlice(f32, aligned), rgb, src_h, src_w, th, tw);
    return .{
        .pixels = buf,
        .h = th,
        .w = tw,
        .tokens = (grid.h / DOWNSAMPLE) * (grid.w / DOWNSAMPLE),
    };
}

pub const VideoPrepared = struct {
    pixels: []u8,
    frames: u32,
    h: u32,
    w: u32,
    /// Language tokens in one tubelet, after the shuffle.
    tokens: u32,
    groups: u32,
};

/// Every frame resized to frame 0's video grid, CLIP-normalized, concatenated
/// as float32 CHW. Frames are added one at a time so a caller can free each
/// decoded frame before the next. The encoder repeats a short last tubelet; this buffer does not.
pub const VideoBuilder = struct {
    buf: []u8,
    frames: u32,
    th: u32,
    tw: u32,

    pub fn init(allocator: std.mem.Allocator, frames: usize, h0: u32, w0: u32) !VideoBuilder {
        if (frames == 0 or frames > std.math.maxInt(u32)) return error.InvalidVisionInput;
        const grid = targetPatchesVideo(h0, w0);
        const th = grid.h * PATCH;
        const tw = grid.w * PATCH;
        return .{ .buf = try allocator.alloc(u8, frames * frameBytes(th, tw)), .frames = @intCast(frames), .th = th, .tw = tw };
    }

    fn frameBytes(th: u32, tw: u32) usize {
        return @as(usize, th) * tw * 3 * @sizeOf(f32);
    }

    pub fn add(self: *VideoBuilder, allocator: std.mem.Allocator, i: usize, rgb: []const u8, h: u32, w: u32) !void {
        const fb = frameBytes(self.th, self.tw);
        const bytes: []align(@alignOf(f32)) u8 = @alignCast(self.buf[i * fb ..][0..fb]);
        try resizeClipChw(allocator, std.mem.bytesAsSlice(f32, bytes), rgb, h, w, self.th, self.tw);
    }

    /// Hands `buf` to the result; on error the caller frees it.
    pub fn finish(self: VideoBuilder) !VideoPrepared {
        const geom = videoGeometry(self.frames, self.th, self.tw) orelse return error.InvalidVisionInput;
        return .{ .pixels = self.buf, .frames = geom.frames, .h = self.th, .w = self.tw, .tokens = geom.spatial, .groups = geom.groups };
    }
};

pub fn prepareVideo(allocator: std.mem.Allocator, rgbs: []const []const u8, hs: []const u32, ws: []const u32) !VideoPrepared {
    if (rgbs.len == 0 or rgbs.len != hs.len or rgbs.len != ws.len) return error.InvalidVisionInput;
    var vb = try VideoBuilder.init(allocator, rgbs.len, hs[0], ws[0]);
    errdefer allocator.free(vb.buf);
    for (rgbs, hs, ws, 0..) |rgb, h, w, i| try vb.add(allocator, i, rgb, h, w);
    return vb.finish();
}

const Block = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fc1_w: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
};

const Axis = struct {
    lo: []i32,
    hi: []i32,
    frac: []f64,

    fn deinit(self: *Axis, a: std.mem.Allocator) void {
        a.free(self.lo);
        a.free(self.hi);
        a.free(self.frac);
    }
};

fn bilinearAxis(a: std.mem.Allocator, len: u32, side: u32) !Axis {
    const lo = try a.alloc(i32, len);
    errdefer a.free(lo);
    const hi = try a.alloc(i32, len);
    errdefer a.free(hi);
    const frac = try a.alloc(f64, len);
    const last: i32 = @intCast(side - 1);
    const scale = @as(f64, @floatFromInt(side)) / @as(f64, @floatFromInt(len));
    for (0..len) |i| {
        const g = (@as(f64, @floatFromInt(i)) + 0.5) * scale - 0.5;
        const fl = @floor(g);
        const fi: i32 = @intFromFloat(fl);
        frac[i] = g - fl;
        lo[i] = std.math.clamp(fi, 0, last);
        hi[i] = std.math.clamp(fi + 1, 0, last);
    }
    return .{ .lo = lo, .hi = hi, .frac = frac };
}

fn must(weights: *const Weights, buf: *[320]u8, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return error.MissingVisionWeights;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingVisionWeights;
    };
}

fn loadProjectorLinear(weights: *const Weights, base: []const u8, in_dim: u32) !glm_vision.Lin {
    var buf: [320]u8 = undefined;
    var linear = glm_vision.Lin{ .w = try must(weights, &buf, "{s}.weight", .{base}) };
    const shape = mlx.getShape(linear.w);
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidVisionWeights;
    const scales_name = std.fmt.bufPrint(&buf, "{s}.scales", .{base}) catch return error.InvalidVisionWeights;
    if (weights.get(scales_name)) |scales| {
        linear.scales = scales;
        const bias_name = std.fmt.bufPrint(&buf, "{s}.biases", .{base}) catch return error.InvalidVisionWeights;
        linear.biases = weights.get(bias_name) orelse .{ .ctx = null };
        const scale_shape = mlx.getShape(scales);
        if (mlx.mlx_array_dtype(linear.w) != .uint32 or scale_shape.len != 2 or scale_shape[0] != shape[0]) return error.InvalidVisionWeights;
        if (linear.biases.ctx != null and !std.mem.eql(c_int, scale_shape, mlx.getShape(linear.biases))) return error.InvalidVisionWeights;
        linear.qp = params.quantParamsFromGeometry(linear.w, scales, linear.biases.ctx != null, in_dim) orelse return error.InvalidVisionWeights;
        if (linear.qp.?.mode == .affine and linear.biases.ctx == null) return error.InvalidVisionWeights;
    } else if (shape[1] != in_dim or mlx.mlx_array_dtype(linear.w) == .uint32) {
        return error.InvalidVisionWeights;
    }
    return linear;
}

fn patchGrid(height: u32, width: u32) !struct { u32, u32 } {
    if (height % PATCH != 0 or width % PATCH != 0) return error.InvalidVisionInput;
    const rows = height / PATCH;
    const cols = width / PATCH;
    if (rows % DOWNSAMPLE != 0 or cols % DOWNSAMPLE != 0 or rows == 0 or cols == 0) return error.InvalidVisionInput;
    return .{ rows, cols };
}

fn loadVideoEmbedder(weights: *const Weights, hidden: u32) mlx.mlx_array {
    const name = "vision_model.radio_model.model.patch_generator.video_embedder.weight";
    const w = weights.get(name) orelse {
        log.info("RADIO video embedder absent; still images only\n", .{});
        return .{ .ctx = null };
    };
    const sh = mlx.getShape(w);
    const expect: c_int = @intCast(TEMPORAL * 3 * PATCH * PATCH);
    if (sh.len != 2 or sh[0] != @as(c_int, @intCast(hidden)) or sh[1] != expect) {
        log.err("RADIO video embedder shape does not match the temporal pack. Video is off; still images stay on.\n", .{});
        return .{ .ctx = null };
    }
    log.info("RADIO video embedder: in={d} temporal={d}\n", .{ expect, TEMPORAL });
    return w;
}

pub const RadioVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    ln_eps: f32,
    prefix: u32,
    pos_side: u32,
    out_hidden: u32,
    pos_owned: bool,
    patch_w: mlx.mlx_array,
    /// Bias-free `video_embedder`. Null when the checkpoint has no video tower;
    /// still images keep working.
    video_w: mlx.mlx_array,
    pos_table: mlx.mlx_array,
    cls: mlx.mlx_array,
    blocks: []Block,
    final_norm_w: mlx.mlx_array,
    final_norm_b: mlx.mlx_array,
    proj_norm: mlx.mlx_array,
    proj_fc1: glm_vision.Lin,
    proj_fc2: glm_vision.Lin,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !RadioVision {
        const s = mlx.gpuStream();
        errdefer _ = mlx.mlx_stream_free(s);
        var buf: [320]u8 = undefined;
        var depth: usize = 0;
        while (depth < 64) : (depth += 1) {
            const name = std.fmt.bufPrint(&buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.weight", .{depth}) catch break;
            if (weights.get(name) == null) break;
        }
        if (depth == 0) {
            log.err("MISSING WEIGHT: vision_model.radio_model.model.blocks.0.attn.qkv.weight\n", .{});
            return error.MissingVisionWeights;
        }
        const qkv0 = try must(weights, &buf, "vision_model.radio_model.model.blocks.0.attn.qkv.weight", .{});
        const qshape = mlx.getShape(qkv0);
        if (qshape.len != 2 or qshape[0] <= 0 or qshape[1] <= 0) return error.InvalidVisionWeights;
        const hidden: u32 = @intCast(qshape[1]);
        const heads = config.radio_num_heads;
        if (heads == 0 or @as(u32, @intCast(qshape[0])) != hidden * 3 or hidden % heads != 0) return error.InvalidVisionWeights;

        var blocks = try allocator.alloc(Block, depth);
        errdefer allocator.free(blocks);
        for (0..depth) |i| {
            blocks[i] = .{
                .norm1_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm1.weight", .{i}),
                .norm1_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm1.bias", .{i}),
                .norm2_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm2.weight", .{i}),
                .norm2_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.norm2.bias", .{i}),
                .qkv_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.weight", .{i}),
                .qkv_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.qkv.bias", .{i}),
                .proj_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.proj.weight", .{i}),
                .proj_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.attn.proj.bias", .{i}),
                .fc1_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc1.weight", .{i}),
                .fc1_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc1.bias", .{i}),
                .fc2_w = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc2.weight", .{i}),
                .fc2_b = try must(weights, &buf, "vision_model.radio_model.model.blocks.{d}.mlp.fc2.bias", .{i}),
            };
        }

        const pos_raw = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.pos_embed", .{});
        const ps = mlx.getShape(pos_raw);
        if (!((ps.len == 2 and ps[1] == hidden) or (ps.len == 3 and ps[0] == 1 and ps[2] == hidden))) return error.InvalidVisionWeights;
        var pos_table = pos_raw;
        var pos_owned = false;
        if (ps.len == 3 and ps[0] == 1) {
            var flat = mlx.mlx_array_new();
            const shape = [_]c_int{ ps[1], ps[2] };
            try mlx.check(mlx.mlx_reshape(&flat, pos_raw, &shape, 2, s));
            pos_table = flat;
            pos_owned = true;
        }
        errdefer if (pos_owned) {
            _ = mlx.mlx_array_free(pos_table);
        };
        const npos: u32 = @intCast(mlx.getShape(pos_table)[0]);
        const side_f = @sqrt(@as(f64, @floatFromInt(npos)));
        const side: u32 = @intFromFloat(@round(side_f));
        if (side == 0 or side * side != npos) return error.InvalidVisionWeights;

        const cls = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.cls_token.token", .{});
        const cls_shape = mlx.getShape(cls);
        if (cls_shape.len != 2 or cls_shape[0] <= 0 or cls_shape[1] != hidden) return error.InvalidVisionWeights;
        const nested = weights.get("mlp1.layers.0.weight") != null;
        const proj_norm = try must(weights, &buf, "{s}.0.weight", .{if (nested) "mlp1.layers" else "mlp1"});
        const norm_shape = mlx.getShape(proj_norm);
        if (norm_shape.len != 1 or norm_shape[0] != hidden * DOWNSAMPLE * DOWNSAMPLE) return error.InvalidVisionWeights;
        const fc1 = try loadProjectorLinear(weights, if (nested) "mlp1.layers.1" else "mlp1.1", hidden * DOWNSAMPLE * DOWNSAMPLE);
        const fc1_shape = mlx.getShape(fc1.w);
        const fc2 = try loadProjectorLinear(weights, if (nested) "mlp1.layers.3" else "mlp1.3", @intCast(fc1_shape[0]));
        const fc2_shape = mlx.getShape(fc2.w);
        var final_w = mlx.mlx_array{ .ctx = null };
        var final_b = mlx.mlx_array{ .ctx = null };
        if (config.radio_final_layernorm) {
            final_w = try must(weights, &buf, "vision_projector.vision_final_layernorm.weight", .{});
            final_b = try must(weights, &buf, "vision_projector.vision_final_layernorm.bias", .{});
            for ([_]mlx.mlx_array{ final_w, final_b }) |value| {
                const shape = mlx.getShape(value);
                if (shape.len != 1 or shape[0] != hidden) return error.InvalidVisionWeights;
            }
        }

        log.info("Vision encoder: RADIO (depth={d}, hidden={d}, heads={d}, prefix={d}, pos={d}, out={d}, patch cap={d})\n", .{
            depth, hidden, heads, cls_shape[0], side, fc2_shape[0], ENGINE_MAX_PATCHES,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = heads,
            .head_dim = hidden / heads,
            .ln_eps = config.radio_ln_eps,
            .prefix = @intCast(cls_shape[0]),
            .pos_side = side,
            .out_hidden = @intCast(fc2_shape[0]),
            .pos_owned = pos_owned,
            .patch_w = try must(weights, &buf, "vision_model.radio_model.model.patch_generator.embedder.weight", .{}),
            .video_w = loadVideoEmbedder(weights, hidden),
            .pos_table = pos_table,
            .cls = cls,
            .blocks = blocks,
            .final_norm_w = final_w,
            .final_norm_b = final_b,
            .proj_norm = proj_norm,
            .proj_fc1 = fc1,
            .proj_fc2 = fc2,
        };
    }

    pub fn deinit(self: *RadioVision) void {
        if (self.pos_owned) _ = mlx.mlx_array_free(self.pos_table);
        self.allocator.free(self.blocks);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// pixels `[1, 3, H, W]` float32, H and W divisible by patch and by the shuffle.
    /// Returns `[1, (H/patch/2)*(W/patch/2), out_hidden]`.
    pub fn forward(self: *RadioVision, pixels: mlx.mlx_array) !mlx.mlx_array {
        const shape = mlx.getShape(pixels);
        if (shape.len != 4 or shape[0] != 1 or shape[1] != 3) return error.InvalidVisionInput;
        const height: u32 = @intCast(shape[2]);
        const width: u32 = @intCast(shape[3]);
        const rows, const cols = try patchGrid(height, width);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, pixels, mlx.mlx_array_dtype(self.patch_w), self.s));
        defer _ = mlx.mlx_array_free(x);
        return self.forwardBody(x, rows, cols, 3, self.patch_w);
    }

    /// frames `[N, 3, H, W]` float32, one clip already at the shared video size.
    /// A short last tubelet repeats the last real frame. Returns
    /// `[1, groups * spatial, out_hidden]` before EVS.
    pub fn forwardVideo(self: *RadioVision, frames: mlx.mlx_array) !mlx.mlx_array {
        if (self.video_w.ctx == null) return error.MissingVisionWeights;
        const shape = mlx.getShape(frames);
        if (shape.len != 4 or shape[1] != 3 or shape[0] <= 0) return error.InvalidVisionInput;
        const n: u32 = @intCast(shape[0]);
        const height: u32 = @intCast(shape[2]);
        const width: u32 = @intCast(shape[3]);
        const rows, const cols = try patchGrid(height, width);
        const groups: usize = (n + TEMPORAL - 1) / TEMPORAL;

        var parts = std.ArrayList(mlx.mlx_array).empty;
        defer {
            for (parts.items) |p| _ = mlx.mlx_array_free(p);
            parts.deinit(self.allocator);
        }
        for (0..groups) |g| {
            const tube = try self.packTubelet(frames, @intCast(g), n, height, width);
            defer _ = mlx.mlx_array_free(tube);
            var x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&x, tube, mlx.mlx_array_dtype(self.video_w), self.s));
            const tokens = self.forwardBody(x, rows, cols, TEMPORAL * 3, self.video_w) catch |err| {
                _ = mlx.mlx_array_free(x);
                return err;
            };
            _ = mlx.mlx_array_free(x);
            // One tubelet at a time. The tower graph is large and nothing in it is reused.
            try mlx.check(mlx.mlx_array_eval(tokens));
            parts.append(self.allocator, tokens) catch |err| {
                _ = mlx.mlx_array_free(tokens);
                return err;
            };
        }
        if (parts.items.len == 1) {
            const only = parts.items[0];
            parts.items[0] = mlx.mlx_array_new();
            return only;
        }
        const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        return out;
    }

    /// EVS keep-mask for the rows `[row_off, row_off + groups*spatial)` of
    /// `emb` (`[1, rows, hidden]`). The mask is the full pre-prune length.
    /// `rate <= 0` keeps every row and does not read the tensor.
    pub fn videoKeepMask(self: *RadioVision, allocator: std.mem.Allocator, emb: mlx.mlx_array, row_off: usize, groups: u32, spatial: u32, rate: f64) ![]bool {
        const full: usize = @as(usize, groups) * spatial;
        if (!(rate > 0)) {
            const mask = try allocator.alloc(bool, full);
            @memset(mask, true);
            return mask;
        }
        const shape = mlx.getShape(emb);
        if (shape.len != 3 or shape[0] != 1) return error.InvalidVisionInput;
        const hidden: usize = @intCast(shape[2]);
        const total: usize = @intCast(shape[1]);
        if (hidden == 0 or row_off + full > total) return error.InvalidVisionInput;

        const start = [_]c_int{ 0, @intCast(row_off), 0 };
        const stop = [_]c_int{ 1, @intCast(row_off + full), @intCast(hidden) };
        const strides = [_]c_int{ 1, 1, 1 };
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_slice(&sl, emb, &start, 3, &stop, 3, &strides, 3, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_astype(&flat, sl, .float32, self.s));
        try mlx.check(mlx.mlx_array_eval(flat));
        if (mlx.mlx_array_size(flat) != full * hidden) return error.InvalidVisionInput;
        const ptr = mlx.mlx_array_data_float32(flat) orelse return error.InvalidVisionInput;
        const host = try allocator.alloc(f32, full * hidden);
        defer allocator.free(host);
        @memcpy(host, ptr[0 .. full * hidden]);
        const scores = try allocator.alloc(f32, full);
        defer allocator.free(scores);
        videoDissimilarity(host, groups, spatial, hidden, scores);
        return evsRetentionMask(allocator, scores, evsKeepCount(groups, spatial, rate));
    }

    fn forwardBody(self: *RadioVision, x_in: mlx.mlx_array, rows: u32, cols: u32, channels: u32, proj: mlx.mlx_array) !mlx.mlx_array {
        var x = try self.patchify(x_in, rows, cols, channels, proj);
        errdefer _ = mlx.mlx_array_free(x);
        {
            const pos = try self.posEmbed(rows, cols);
            defer _ = mlx.mlx_array_free(pos);
            var summed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&summed, x, pos, self.s));
            glm_vision.replace(&x, summed);
        }
        glm_vision.replace(&x, try self.prependCls(x));

        for (self.blocks) |blk| {
            const normed = try self.layerNorm(x, blk.norm1_w, blk.norm1_b);
            defer _ = mlx.mlx_array_free(normed);
            const attn = try self.attention(normed, blk);
            defer _ = mlx.mlx_array_free(attn);
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
            glm_vision.replace(&x, h);

            const normed2 = try self.layerNorm(x, blk.norm2_w, blk.norm2_b);
            defer _ = mlx.mlx_array_free(normed2);
            const up = try self.dense(normed2, blk.fc1_w, blk.fc1_b);
            defer _ = mlx.mlx_array_free(up);
            const act = try glm_vision.geluErf(up, self.s);
            defer _ = mlx.mlx_array_free(act);
            const down = try self.dense(act, blk.fc2_w, blk.fc2_b);
            defer _ = mlx.mlx_array_free(down);
            var h2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h2, x, down, self.s));
            glm_vision.replace(&x, h2);
        }

        glm_vision.replace(&x, try self.dropPrefix(x));
        if (self.final_norm_w.ctx != null) glm_vision.replace(&x, try self.layerNorm(x, self.final_norm_w, self.final_norm_b));
        glm_vision.replace(&x, try self.pixelShuffle(x, rows, cols));
        glm_vision.replace(&x, try self.projector(x));
        return x;
    }

    /// `[1, T*3, H, W]` for tubelet `g`. A group past the last real frame repeats it.
    fn packTubelet(self: *RadioVision, frames: mlx.mlx_array, g: u32, n: u32, height: u32, width: u32) !mlx.mlx_array {
        var slices: [TEMPORAL]mlx.mlx_array = undefined;
        var made: usize = 0;
        errdefer {
            for (slices[0..made]) |a| _ = mlx.mlx_array_free(a);
        }
        for (0..TEMPORAL) |j| {
            const idx: u32 = @min(g * TEMPORAL + @as(u32, @intCast(j)), n - 1);
            const start = [_]c_int{ @intCast(idx), 0, 0, 0 };
            const stop = [_]c_int{ @intCast(idx + 1), 3, @intCast(height), @intCast(width) };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            slices[j] = mlx.mlx_array_new();
            made += 1;
            try mlx.check(mlx.mlx_slice(&slices[j], frames, &start, 4, &stop, 4, &strides, 4, self.s));
        }
        const vec = mlx.mlx_vector_array_new_data(&slices, TEMPORAL);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        for (slices) |a| _ = mlx.mlx_array_free(a);
        made = 0;
        return out;
    }

    fn patchify(self: *RadioVision, x: mlx.mlx_array, rows: u32, cols: u32, channels: u32, proj: mlx.mlx_array) !mlx.mlx_array {
        const ps: c_int = PATCH;
        const ch: c_int = @intCast(channels);
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        const in_shape = [_]c_int{ 1, ch, @intCast(rows), ps, @intCast(cols), ps };
        try mlx.check(mlx.mlx_reshape(&r, x, &in_shape, 6, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        const axes = [_]c_int{ 0, 2, 4, 1, 3, 5 };
        try mlx.check(mlx.mlx_transpose_axes(&t, r, &axes, 6, self.s));
        const flat = [_]c_int{ 1, @intCast(rows * cols), ch * ps * ps };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, t, &flat, 3, self.s));
        const projected = try self.dense(out, proj, .{ .ctx = null });
        _ = mlx.mlx_array_free(out);
        return projected;
    }

    fn prependCls(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        var cls = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cls);
        const shape = [_]c_int{ 1, @intCast(self.prefix), @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&cls, self.cls, &shape, 3, self.s));
        const parts = [_]mlx.mlx_array{ cls, x };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        return out;
    }

    fn dropPrefix(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const start = [_]c_int{ 0, @intCast(self.prefix), 0 };
        const stop = [_]c_int{ shape[0], shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&out, x, &start, 3, &stop, 3, &strides, 3, self.s));
        return out;
    }

    /// Cropped position embedding: bilinear to max(rows, cols), then the top-left crop.
    fn posEmbed(self: *RadioVision, rows: u32, cols: u32) !mlx.mlx_array {
        const a = self.allocator;
        const side = self.pos_side;
        const max_dim = @max(rows, cols);
        var ha = try bilinearAxis(a, max_dim, side);
        defer ha.deinit(a);
        var wa = try bilinearAxis(a, max_dim, side);
        defer wa.deinit(a);

        const n: usize = @as(usize, rows) * cols;
        const indices = try a.alloc(i32, n * 4);
        defer a.free(indices);
        const weights = try a.alloc(f32, n * 4);
        defer a.free(weights);
        var idx: [4][]i32 = undefined;
        var wgt: [4][]f32 = undefined;
        inline for (0..4) |c| {
            idx[c] = indices[c * n ..][0..n];
            wgt[c] = weights[c * n ..][0..n];
        }

        const side_i: i32 = @intCast(side);
        for (0..rows) |r| {
            for (0..cols) |c| {
                const t = r * cols + c;
                const dh = ha.frac[r];
                const dw = wa.frac[c];
                idx[0][t] = ha.lo[r] * side_i + wa.lo[c];
                idx[1][t] = ha.lo[r] * side_i + wa.hi[c];
                idx[2][t] = ha.hi[r] * side_i + wa.lo[c];
                idx[3][t] = ha.hi[r] * side_i + wa.hi[c];
                wgt[0][t] = @floatCast((1 - dh) * (1 - dw));
                wgt[1][t] = @floatCast((1 - dh) * dw);
                wgt[2][t] = @floatCast(dh * (1 - dw));
                wgt[3][t] = @floatCast(dh * dw);
            }
        }

        const wshape = [_]c_int{ @intCast(n), 1 };
        var table = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(table);
        try mlx.check(mlx.mlx_astype(&table, self.pos_table, .float32, self.s));
        var acc: mlx.mlx_array = undefined;
        inline for (0..4) |c| {
            const ix = hostI32(idx[c]);
            defer _ = mlx.mlx_array_free(ix);
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            try mlx.check(mlx.mlx_take_axis(&gathered, table, ix, 0, self.s));
            const wf = mlx.mlx_array_new_data(wgt[c].ptr, &wshape, 2, .float32);
            defer _ = mlx.mlx_array_free(wf);
            var weighted = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&weighted, gathered, wf, self.s));
            if (c == 0) {
                acc = weighted;
            } else {
                var sum = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&sum, acc, weighted, self.s));
                _ = mlx.mlx_array_free(weighted);
                _ = mlx.mlx_array_free(acc);
                acc = sum;
            }
        }
        defer _ = mlx.mlx_array_free(acc);
        var cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, acc, mlx.mlx_array_dtype(self.patch_w), self.s));
        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, @intCast(n), @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&out, cast, &oshape, 3, self.s));
        return out;
    }

    fn attention(self: *RadioVision, x: mlx.mlx_array, blk: Block) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const seq = shape[1];
        const heads: c_int = @intCast(self.heads);
        const hd: c_int = @intCast(self.head_dim);
        const qkv = try self.dense(x, blk.qkv_w, blk.qkv_b);
        defer _ = mlx.mlx_array_free(qkv);
        var parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        try mlx.check(mlx.mlx_split(&parts, qkv, 3, 2, self.s));

        var qkv_h: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer {
            for (qkv_h) |a| _ = mlx.mlx_array_free(a);
        }
        for (0..3) |j| {
            var piece = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(piece);
            try mlx.check(mlx.mlx_vector_array_get(&piece, parts, j));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, piece, &[_]c_int{ 1, seq, heads, hd }, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&qkv_h[j], r, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx, qkv_h[0], qkv_h[1], qkv_h[2], scale, "", none, none, false, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ 1, seq, @intCast(self.hidden) }, 3, self.s));
        return self.dense(flat, blk.proj_w, blk.proj_b);
    }

    fn pixelShuffle(self: *RadioVision, x: mlx.mlx_array, rows: u32, cols: u32) !mlx.mlx_array {
        const c: c_int = @intCast(self.hidden);
        var grid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grid);
        try mlx.check(mlx.mlx_reshape(&grid, x, &[_]c_int{ 1, @intCast(rows), @intCast(cols), c }, 4, self.s));
        var v1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v1);
        try mlx.check(mlx.mlx_reshape(&v1, grid, &[_]c_int{ 1, @intCast(rows), @intCast(cols / 2), c * 2 }, 4, self.s));
        var p1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p1);
        try mlx.check(mlx.mlx_transpose_axes(&p1, v1, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var v2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v2);
        try mlx.check(mlx.mlx_reshape(&v2, p1, &[_]c_int{ 1, @intCast(cols / 2), @intCast(rows / 2), c * 4 }, 4, self.s));
        var p2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p2);
        try mlx.check(mlx.mlx_transpose_axes(&p2, v2, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        const ntok: c_int = @intCast((rows / 2) * (cols / 2));
        var flat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&flat, p2, &[_]c_int{ 1, ntok, c * 4 }, 3, self.s));
        return flat;
    }

    fn projector(self: *RadioVision, x: mlx.mlx_array) !mlx.mlx_array {
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, x, self.proj_norm, PROJECTOR_EPS, self.s));
        const up = try glm_vision.linear(normed, self.proj_fc1, self.s);
        defer _ = mlx.mlx_array_free(up);
        const act = try activations.reluSquaredOp(up, self.s);
        defer _ = mlx.mlx_array_free(act);
        return glm_vision.linear(act, self.proj_fc2, self.s);
    }

    fn layerNorm(self: *RadioVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.ln_eps, self.s));
        return out;
    }

    fn dense(self: *RadioVision, x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
        return glm_vision.linear(x, .{ .w = w, .bias = bias }, self.s);
    }
};

fn hostI32(v: []const i32) mlx.mlx_array {
    const shape = [_]c_int{@intCast(v.len)};
    return mlx.mlx_array_new_data(v.ptr, &shape, 1, .int32);
}

const RadioFixture = struct {
    tmp: std.testing.TmpDir,
    weights: Weights,

    fn open() !RadioFixture {
        return openEmbedded(@embedFile("../../fixtures/radio_tiny.safetensors"));
    }

    fn openEmbedded(bytes: []const u8) !RadioFixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "radio.safetensors", .data = bytes });
        var buf: [1024]u8 = undefined;
        const dir = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
        return .{ .tmp = tmp, .weights = try model_mod.loadWeightsWithVision(std.testing.io, std.testing.allocator, dir) };
    }

    fn deinit(self: *RadioFixture) void {
        self.weights.deinit();
        self.tmp.cleanup();
    }

    fn tensor(self: *const RadioFixture, name: []const u8) !mlx.mlx_array {
        return self.weights.get(name) orelse error.MissingFixtureTensor;
    }
};

fn expectReferenceTensor(got: mlx.mlx_array, want: mlx.mlx_array, s: mlx.mlx_stream, tolerance: f32) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var fp32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fp32);
    try mlx.check(mlx.mlx_astype(&fp32, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(fp32));
    try mlx.check(mlx.mlx_array_eval(want));
    const n = mlx.mlx_array_size(want);
    const actual = mlx.mlx_array_data_float32(fp32).?[0..n];
    const expected = mlx.mlx_array_data_float32(want).?[0..n];
    var worst: f32 = 0;
    for (actual, expected) |x, y| {
        try std.testing.expect(std.math.isFinite(x));
        worst = @max(worst, @abs(x - y));
    }
    std.debug.print("[radio-reference] max|diff|={e}\n", .{worst});
    try std.testing.expect(worst < tolerance);
}

test "radio reference: full image/video forward and EVS on a decoded clip" {
    var fx = try RadioFixture.open();
    defer fx.deinit();
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2 }, &fx.weights);
    defer rv.deinit();
    _ = mlx.mlx_stream_free(rv.s);
    rv.s = mlx.mlx_default_cpu_stream_new();
    try std.testing.expectEqual(@as(u32, 2), rv.heads);
    const image = try rv.forward(try fx.tensor("image_pixels"));
    defer _ = mlx.mlx_array_free(image);
    try expectReferenceTensor(image, try fx.tensor("image_features"), rv.s, 2e-5);
    const video = try rv.forwardVideo(try fx.tensor("video_pixels"));
    defer _ = mlx.mlx_array_free(video);
    try expectReferenceTensor(video, try fx.tensor("video_features"), rv.s, 2e-5);
    const mask = try rv.videoKeepMask(std.testing.allocator, video, 0, 3, 2, 0.3);
    defer std.testing.allocator.free(mask);
    const expected = try fx.tensor("video_mask");
    try mlx.check(mlx.mlx_array_eval(expected));
    try std.testing.expectEqualSlices(bool, mlx.mlx_array_data_bool(expected).?[0..mask.len], mask);
}

test "radio Super reference: native quantized projector and pre-shuffle normalization" {
    var fx = try RadioFixture.openEmbedded(@embedFile("../../fixtures/radio_super_tiny.safetensors"));
    defer fx.deinit();
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2, .radio_final_layernorm = true }, &fx.weights);
    defer rv.deinit();
    try std.testing.expectEqual(@as(u32, 4), rv.proj_fc1.qp.?.bits);
    try std.testing.expectEqual(@as(u32, 64), rv.proj_fc1.qp.?.group_size);
    try std.testing.expectEqual(@as(u32, 4), rv.proj_fc2.qp.?.bits);
    try std.testing.expectEqual(@as(u32, 64), rv.proj_fc2.qp.?.group_size);
    _ = mlx.mlx_stream_free(rv.s);
    rv.s = mlx.mlx_default_cpu_stream_new();
    for ([_]bool{ false, true }) |video| {
        const pixels = try fx.tensor(if (video) "video_pixels" else "image_pixels");
        const got = if (video) try rv.forwardVideo(pixels) else try rv.forward(pixels);
        defer _ = mlx.mlx_array_free(got);
        try expectReferenceTensor(got, try fx.tensor(if (video) "video_features" else "image_features"), rv.s, 2e-5);
    }
}

test "radio Super loader: dense nested projector shares the quantized reference" {
    var fx = try RadioFixture.openEmbedded(@embedFile("../../fixtures/radio_super_tiny.safetensors"));
    defer fx.deinit();
    inline for (.{ "1", "3" }, .{ "projector_linear1_dense", "projector_linear2_dense" }) |layer, reference| {
        const dense = try fx.tensor(reference);
        try mlx.check(mlx.mlx_array_set(fx.weights.map.getPtr("mlp1.layers." ++ layer ++ ".weight").?, dense));
        inline for (.{ "scales", "biases" }) |suffix| {
            const entry = fx.weights.map.fetchRemove("mlp1.layers." ++ layer ++ "." ++ suffix).?;
            fx.weights.allocator.free(entry.key);
            _ = mlx.mlx_array_free(entry.value);
        }
    }
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2, .radio_final_layernorm = true }, &fx.weights);
    defer rv.deinit();
    try std.testing.expect(rv.proj_fc1.qp == null and rv.proj_fc2.qp == null);
    _ = mlx.mlx_stream_free(rv.s);
    rv.s = mlx.mlx_default_cpu_stream_new();
    const got = try rv.forwardVideo(try fx.tensor("video_pixels"));
    defer _ = mlx.mlx_array_free(got);
    try expectReferenceTensor(got, try fx.tensor("video_features"), rv.s, 2e-5);
}

test "radio Super loader: required final norm and quantization packets are complete" {
    var fx = try RadioFixture.openEmbedded(@embedFile("../../fixtures/radio_super_tiny.safetensors"));
    defer fx.deinit();
    const config = ModelConfig{ .radio_num_heads = 2, .radio_final_layernorm = true };
    for ([_][]const u8{ "vision_projector.vision_final_layernorm.bias", "mlp1.layers.1.biases" }) |name| {
        const removed = fx.weights.map.fetchRemove(name).?;
        defer fx.weights.map.put(removed.key, removed.value) catch unreachable;
        if (std.mem.endsWith(u8, name, "layernorm.bias")) {
            try std.testing.expectError(error.MissingVisionWeights, RadioVision.init(std.testing.allocator, config, &fx.weights));
        } else {
            try std.testing.expectError(error.InvalidVisionWeights, RadioVision.init(std.testing.allocator, config, &fx.weights));
        }
    }
    const entry = fx.weights.map.getPtr("mlp1.layers.1.scales").?;
    const original = entry.*;
    entry.* = try fx.tensor("mlp1.layers.3.scales");
    defer entry.* = original;
    try std.testing.expectError(error.InvalidVisionWeights, RadioVision.init(std.testing.allocator, config, &fx.weights));
}

test "radio Super reference: bf16 quantized image/video preserve magnitude" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var fx = try RadioFixture.openEmbedded(@embedFile("../../fixtures/radio_super_tiny.safetensors"));
    defer fx.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var it = fx.weights.map.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "vision_model.") and !std.mem.startsWith(u8, name, "mlp1.") and !std.mem.startsWith(u8, name, "vision_projector.")) continue;
        if (mlx.mlx_array_dtype(entry.value_ptr.*) == .uint32) continue;
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, entry.value_ptr.*, .bfloat16, s));
        _ = mlx.mlx_array_free(entry.value_ptr.*);
        entry.value_ptr.* = cast;
    }
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2, .radio_final_layernorm = true }, &fx.weights);
    defer rv.deinit();
    for ([_]bool{ false, true }) |video| {
        const got = if (video) try rv.forwardVideo(try fx.tensor("video_pixels")) else try rv.forward(try fx.tensor("image_pixels"));
        defer _ = mlx.mlx_array_free(got);
        const want = try fx.tensor(if (video) "video_features" else "image_features");
        const c = try glm_vision.cosineSim(got, want, rv.s);
        const r = try glm_vision.rmsRatio(got, want, rv.s);
        std.debug.print("[radio-Super] bf16 video={any} cosine={d:.7} rms_ratio={d:.6}\n", .{ video, c, r });
        try std.testing.expect(c > 0.9999 and r > 0.98 and r < 1.02);
    }
}

test "radio reference: bf16 image/video serving preserves reference direction and magnitude" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var fx = try RadioFixture.open();
    defer fx.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var it = fx.weights.map.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "vision_model.") and !std.mem.startsWith(u8, name, "mlp1.")) continue;
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, entry.value_ptr.*, .bfloat16, s));
        _ = mlx.mlx_array_free(entry.value_ptr.*);
        entry.value_ptr.* = cast;
    }
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2 }, &fx.weights);
    defer rv.deinit();
    for ([_]bool{ false, true }) |video| {
        const got = if (video) try rv.forwardVideo(try fx.tensor("video_pixels")) else try rv.forward(try fx.tensor("image_pixels"));
        defer _ = mlx.mlx_array_free(got);
        const want = try fx.tensor(if (video) "video_features" else "image_features");
        const c = try glm_vision.cosineSim(got, want, rv.s);
        const r = try glm_vision.rmsRatio(got, want, rv.s);
        std.debug.print("[radio-reference] bf16 video={any} cosine={d:.7} rms_ratio={d:.6}\n", .{ video, c, r });
        try std.testing.expect(c > 0.9999 and r > 0.98 and r < 1.02);
    }
}

test "radio reference: cropped position interpolation clamps boundaries and keeps fp32 precision" {
    var fx = try RadioFixture.open();
    defer fx.deinit();
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2 }, &fx.weights);
    defer rv.deinit();
    const pos = try rv.posEmbed(2, 4);
    defer _ = mlx.mlx_array_free(pos);
    try expectReferenceTensor(pos, try fx.tensor("image_positions"), rv.s, 1e-7);
}

test "radio reference: float bicubic preprocessing matches torch antialias without a uint8 boundary" {
    var fx = try RadioFixture.open();
    defer fx.deinit();
    const rgb = try fx.tensor("resize_rgb");
    try mlx.check(mlx.mlx_array_eval(rgb));
    const input = mlx.mlx_array_data_uint8(rgb).?[0 .. 8 * 9 * 3];
    for ([_]struct { name: []const u8, h: u32, w: u32 }{
        .{ .name = "resize_down", .h = 3, .w = 4 },
        .{ .name = "resize_up", .h = 11, .w = 13 },
    }) |case| {
        const n: usize = @as(usize, case.h) * case.w;
        const actual = try std.testing.allocator.alloc(f32, n * 3);
        defer std.testing.allocator.free(actual);
        try resizeClipChw(std.testing.allocator, actual, input, 8, 9, case.h, case.w);
        const ref = try fx.tensor(case.name);
        try mlx.check(mlx.mlx_array_eval(ref));
        const expected = mlx.mlx_array_data_float32(ref).?[0..actual.len];
        var worst: f32 = 0;
        for (actual, expected, 0..) |x, y, i| {
            const want = (y / 255 - MEAN[i / n]) / STD[i / n];
            worst = @max(worst, @abs(x - want));
        }
        std.debug.print("[radio-reference] {s} normalized pixels max|diff|={e}\n", .{ case.name, worst });
        try std.testing.expect(worst < 4e-6);
    }
}

test "radio reference: EVS keeps the reference token count for custom pruning rates" {
    try std.testing.expectEqual(@as(u32, 18), evsKeepCount(5, 4, 0.1));
    try std.testing.expectEqual(@as(u32, 14), evsKeepCount(5, 4, 0.3));
    try std.testing.expectEqual(@as(u32, 6), evsKeepCount(5, 4, 0.7));
    var fx = try RadioFixture.open();
    defer fx.deinit();
    var rv = try RadioVision.init(std.testing.allocator, .{ .radio_num_heads = 2 }, &fx.weights);
    defer rv.deinit();
    for ([_]struct { name: []const u8, rate: f64 }{
        .{ .name = "evs_mask_10", .rate = 0.1 },
        .{ .name = "evs_mask_30", .rate = 0.3 },
        .{ .name = "evs_mask_70", .rate = 0.7 },
    }) |case| {
        const mask = try rv.videoKeepMask(std.testing.allocator, try fx.tensor("evs_features"), 0, 5, 4, case.rate);
        defer std.testing.allocator.free(mask);
        const expected = try fx.tensor(case.name);
        try mlx.check(mlx.mlx_array_eval(expected));
        try std.testing.expectEqualSlices(bool, mlx.mlx_array_data_bool(expected).?[0..mask.len], mask);
    }
}

test "radio EVS cosine clamps each norm to the reference epsilon" {
    const embeddings = [_]f32{ 1e-10, 0, 1e-10, 0 };
    var scores: [2]f32 = undefined;
    videoDissimilarity(&embeddings, 2, 1, 2, &scores);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9999), scores[1], 1e-6);
}

test "radio full checkpoint: image/video embeddings and EVS match the official fp32 oracle" {
    const model_raw = std.c.getenv("RADIO_VISION_MODEL") orelse return error.SkipZigTest;
    const fixture_raw = std.c.getenv("RADIO_VISION_FIXTURE") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const model_dir = std.mem.span(model_raw);
    var source = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer source.close(io);
    const json = source.readFileAlloc(io, "config.hf.json", a, .limited(1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => try source.readFileAlloc(io, "config.json", a, .limited(1024 * 1024)),
        else => return err,
    };
    defer a.free(json);
    var config = try model_mod.parseConfigFromJson(a, json);
    defer config.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [1024]u8 = undefined;
    const tower_dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const index_json = try source.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024));
    defer a.free(index_json);
    const index = try std.json.parseFromSlice(std.json.Value, a, index_json, .{});
    defer index.deinit();
    var shards = std.StringHashMap(void).init(a);
    defer shards.deinit();
    var entries = index.value.object.get("weight_map").?.object.iterator();
    while (entries.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, key, "vision_model.") and !std.mem.startsWith(u8, key, "mlp1.") and !std.mem.startsWith(u8, key, "vision_projector.")) continue;
        try shards.put(entry.value_ptr.string, {});
    }
    var names = shards.keyIterator();
    while (names.next()) |name| {
        const from = try std.fmt.allocPrint(a, "{s}/{s}", .{ model_dir, name.* });
        defer a.free(from);
        const to = try std.fmt.allocPrint(a, "{s}/{s}", .{ tower_dir, name.* });
        defer a.free(to);
        try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
    }
    var weights = try model_mod.loadWeightsWithVision(io, a, tower_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, std.mem.span(fixture_raw));
    defer fx.deinit();
    var rv = try RadioVision.init(a, config, &weights);
    defer rv.deinit();
    const rgb_arr = fx.get("video_rgb").?;
    try mlx.check(mlx.mlx_array_eval(rgb_arr));
    const rgb_shape = mlx.getShape(rgb_arr);
    const n: usize = @intCast(rgb_shape[0]);
    const height: u32 = @intCast(rgb_shape[1]);
    const width: u32 = @intCast(rgb_shape[2]);
    const frame_bytes: usize = @as(usize, height) * width * 3;
    const raw_rgb = mlx.mlx_array_data_uint8(rgb_arr).?[0 .. frame_bytes * n];
    var rgbs: [5][]const u8 = undefined;
    const hs: [5]u32 = @splat(height);
    const ws: [5]u32 = @splat(width);
    try std.testing.expectEqual(rgbs.len, n);
    for (&rgbs, 0..) |*rgb, i| rgb.* = raw_rgb[i * frame_bytes ..][0..frame_bytes];
    const image_px = try prepareImage(a, rgbs[0], height, width);
    defer a.free(image_px.pixels);
    const video_px = try prepareVideo(a, &rgbs, &hs, &ws);
    defer a.free(video_px.pixels);
    for ([_]bool{ false, true }) |video| {
        const bytes = if (video) video_px.pixels else image_px.pixels;
        const expected = fx.get(if (video) "video_pixels" else "image_pixels").?;
        const shape = mlx.getShape(expected);
        const pixels = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(pixels);
        try expectReferenceTensor(pixels, expected, rv.s, 2e-5);
    }
    for ([_]bool{ false, true }) |video| {
        const pixels = fx.get(if (video) "video_pixels" else "image_pixels").?;
        const got = if (video) try rv.forwardVideo(pixels) else try rv.forward(pixels);
        defer _ = mlx.mlx_array_free(got);
        const want = fx.get(if (video) "video_features" else "image_features").?;
        const c = try glm_vision.cosineSim(got, want, rv.s);
        const r = try glm_vision.rmsRatio(got, want, rv.s);
        std.debug.print("[radio-full] video={any} tokens={d} cosine={d:.7} rms_ratio={d:.6}\n", .{ video, mlx.getShape(got)[1], c, r });
        try std.testing.expect(c > 0.99 and r > 0.97 and r < 1.03);
        if (video) {
            const shape = mlx.getShape(pixels);
            const geom = videoGeometry(@intCast(shape[0]), @intCast(shape[2]), @intCast(shape[3])).?;
            const mask = try rv.videoKeepMask(a, want, 0, geom.groups, geom.spatial, 0.3);
            defer a.free(mask);
            const expected = fx.get("video_mask").?;
            try mlx.check(mlx.mlx_array_eval(expected));
            try std.testing.expectEqualSlices(bool, mlx.mlx_array_data_bool(expected).?[0..mask.len], mask);
        }
    }
}

test "radio targetPatches matches the processor on a 512 square" {
    // 512/16 is an integer. Python round(32.5) is 32, so the grid stays 32×32
    // and the shuffle yields 256 language tokens. The engine cap is above the minimum.
    const g = targetPatches(512, 512, ENGINE_MAX_PATCHES);
    try std.testing.expectEqual(@as(u32, 32), g.w);
    try std.testing.expectEqual(@as(u32, 32), g.h);
}

test "radio targetPatches keeps a wide image inside the patch cap" {
    const g = targetPatches(3000, 4000, ENGINE_MAX_PATCHES);
    try std.testing.expect(g.w % 2 == 0 and g.h % 2 == 0);
    try std.testing.expect(g.w * g.h <= ENGINE_MAX_PATCHES);
    try std.testing.expect(g.w * g.h >= MIN_PATCHES);
}

test "radio targetPatchesVideo stays within the patch budget at extreme aspect ratios" {
    for ([_][2]u32{ .{ 64, 4000 }, .{ 1, 10000 }, .{ 10000, 1 } }) |hw| {
        const g = targetPatchesVideo(hw[0], hw[1]);
        try std.testing.expect(g.w * g.h <= VIDEO_TARGET_PATCHES);
        try std.testing.expect(g.w % DOWNSAMPLE == 0 and g.h % DOWNSAMPLE == 0);
    }
}

test "radio targetPatchesVideo snaps 16:9 and 9:16 onto the 1024-patch budget" {
    const wide = targetPatchesVideo(1080, 1920);
    try std.testing.expectEqual(@as(u32, 42), wide.w);
    try std.testing.expectEqual(@as(u32, 24), wide.h);
    const tall = targetPatchesVideo(1920, 1080);
    try std.testing.expectEqual(@as(u32, 24), tall.w);
    try std.testing.expectEqual(@as(u32, 42), tall.h);
    const square = targetPatchesVideo(16, 16);
    try std.testing.expectEqual(@as(u32, 32), square.w);
    try std.testing.expectEqual(@as(u32, 32), square.h);
    // 640×480 (4:3) snaps down to 36×28 patches: 38×28 would pass 1024.
    const sd = targetPatchesVideo(480, 640);
    try std.testing.expectEqual(@as(u32, 36), sd.w);
    try std.testing.expectEqual(@as(u32, 28), sd.h);
    const geom = videoGeometry(3, wide.h * PATCH, wide.w * PATCH).?;
    try std.testing.expectEqual(@as(u32, 12), geom.token_h);
    try std.testing.expectEqual(@as(u32, 21), geom.token_w);
    try std.testing.expectEqual(@as(u32, 252), geom.spatial);
    try std.testing.expectEqual(@as(u32, 2), geom.groups);
}

test "radio video labels omit a padded frame and separate tubelets with a newline" {
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("Frame 1 and frame 2: ", videoGroupSegment(&buf, 0, 4));
    try std.testing.expectEqualStrings("\nFrame 3 and frame 4: ", videoGroupSegment(&buf, 1, 4));
    try std.testing.expectEqualStrings("\nFrame 3: ", videoGroupSegment(&buf, 1, 3));
    try std.testing.expectEqualStrings("Frame 1: ", videoGroupSegment(&buf, 0, 1));
}

test "radio EVS keeps the first tubelet and the most dissimilar later tokens" {
    // Two tubelets: 30% of the clip is less than one tubelet, so only the first survives.
    try std.testing.expectEqual(@as(u32, 4), evsKeepCount(2, 4, 0.7));
    // Ten tubelets: int(10 * 10 * 0.3) = 30, which is above one tubelet.
    try std.testing.expectEqual(@as(u32, 30), evsKeepCount(10, 10, 0.7));
    try std.testing.expectEqual(@as(u32, 8), evsKeepCount(2, 4, 0));

    const embeds = [_]f32{
        1, 0, // tubelet 0, token 0
        0, 1, // identical to the next tubelet's token 0? token 1 of tubelet 0
        1, 0, // tubelet 1 token 0, identical to tubelet 0 token 0 → dissimilarity 0
        1, 0, // tubelet 1 token 1, orthogonal to (0, 1) → dissimilarity 1
    };
    var scores: [4]f32 = undefined;
    videoDissimilarity(&embeds, 2, 2, 2, &scores);
    try std.testing.expectEqual(@as(f32, 255), scores[0]);
    try std.testing.expectEqual(@as(f32, 255), scores[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0), scores[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), scores[3], 1e-6);

    const hand = [_]f32{ 255, 255, 255, 255, 0.1, 0.9, 0.2, 0.8 };
    const mask = try evsRetentionMask(std.testing.allocator, &hand, 6);
    defer std.testing.allocator.free(mask);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, false, true, false, true }, mask);
}
