//! Interleaved M-RoPE host math for Qwen3.5/3.6 vision-language models.
//!
//! Faithful port of mlx-vlm's `rope_utils._interleaved_position_selector` and
//! `Qwen3_5Model.get_rope_index` (mlx_vlm/models/qwen3_5/language.py). All math
//! here is pure host integer/float work — no MLX — so it is hermetically
//! testable. The text trunk (src/transformer.zig) consumes:
//!   * `getRopeIndex` → per-token 3D position ids (t,h,w) + the decode delta, and
//!   * `interleavedSelector` + `buildCosSin` → the prefill cos/sin tables that
//!     replace the scalar `mlx_fast_rope` on image requests.
//!
//! For TEXT tokens t==h==w, so interleaved M-RoPE collapses to ordinary partial
//! RoPE; the 3D divergence only happens at image tokens (prefill only). See the
//! plan and CLAUDE.md "M-RoPE" notes.

const std = @import("std");

/// Full patch grid of one image BEFORE spatial merge: h = H/patch, w = W/patch,
/// t = temporal frames (1 for a still image). The number of LLM image-pad tokens
/// the grid expands to is `t * (h/merge) * (w/merge)`.
pub const ImageGrid = struct { t: u32, h: u32, w: u32 };

/// Result of `getRopeIndex`: `pos[axis][token]` for axis 0=t, 1=h, 2=w, plus the
/// decode offset `delta = max(pos)+1 - seq_len` (mrope_position_deltas).
pub const RopeIndex = struct {
    pos: [3][]i32,
    delta: i32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RopeIndex) void {
        for (self.pos) |p| self.allocator.free(p);
    }
};

/// Borrowed view of a flat axis-major M-RoPE position table.
///
/// `base` maps a cache-relative position to its absolute position in `pos`.
/// The trunk uses `base = 0`. A speculative head may retain only a suffix of
/// the prompt in its own KV cache, so its cache position 0 can correspond to a
/// later absolute prompt position.
pub const PositionContext = struct {
    pos: []const i32,
    total: usize,
    delta: i32,
    base: usize = 0,

    pub fn absolutePosition(self: PositionContext, relative: usize) usize {
        return self.base + relative;
    }

    /// Return one axis' position id. Positions inside the prompt use the
    /// explicit 3-D table; generated text beyond it has t=h=w and follows the
    /// scalar decode position `absolute + delta`.
    pub fn axisPosition(self: PositionContext, axis: usize, relative: usize) i32 {
        std.debug.assert(axis < 3);
        std.debug.assert(self.pos.len >= 3 * self.total);
        const absolute = self.absolutePosition(relative);
        if (absolute < self.total) return self.pos[axis * self.total + absolute];
        return @as(i32, @intCast(absolute)) + self.delta;
    }
};

/// Map each of the `freq_dim` rotary frequencies to a position axis (0=t,1=h,2=w)
/// for the INTERLEAVED scheme. Mirrors `_interleaved_position_selector`: axis 1
/// (h) claims indices 1,4,7,… and axis 2 (w) claims 2,5,8,…, each bounded by
/// `section*3` and the freq count; everything else stays axis 0 (t).
pub fn interleavedSelector(sel: []u8, mrope_section: [3]u32) void {
    @memset(sel, 0);
    const freq_dim = sel.len;
    inline for ([_]usize{ 1, 2 }) |dim| {
        const limit = @min(@as(usize, mrope_section[dim]) * 3, freq_dim);
        var idx: usize = dim; // offset == dim (1 for h, 2 for w)
        while (idx < limit) : (idx += 3) sel[idx] = @intCast(dim);
    }
}

/// Faithful single-sequence, full-attention-mask port of `get_rope_index` for
/// IMAGE and VIDEO inputs. `tokens` is the POST-EXPANSION id sequence (each
/// image/video's pad run already expanded to its merged-grid token count).
/// `images`/`videos` list the full patch grid per block in DOCUMENT order
/// within their own modality — the two lists are walked independently, and
/// blocks are consumed in the order their markers appear in `tokens` (mirrors
/// `language.py`'s `ed_image < ed_video` pick-first-marker loop: a video grid's
/// `t` is its number of TEMPORAL PATCHES, not divided by `merge`; h/w are).
/// Returns 3×seq position ids (caller owns via `RopeIndex.deinit`) and the
/// decode delta.
pub fn getRopeIndex(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    images: []const ImageGrid,
    videos: []const ImageGrid,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    merge: u32,
) !RopeIndex {
    const seq = tokens.len;
    var rows: [3]std.ArrayList(i32) = .{ .empty, .empty, .empty };
    errdefer for (&rows) |*r| r.deinit(allocator);

    // Running max position across all emitted segments. Because each new segment
    // starts at (previous max + 1) and only increases, this equals the reference's
    // `llm_pos_ids_list[-1].max()`.
    var last_max: i32 = -1;
    var st: usize = 0;
    var image_index: usize = 0;
    var video_index: usize = 0;

    // *_nums = number of vision_start tokens immediately followed by that
    // modality's pad token.
    var image_nums: usize = 0;
    var video_nums: usize = 0;
    {
        var i: usize = 0;
        while (i + 1 < seq) : (i += 1) {
            if (tokens[i] == vision_start_token_id) {
                if (tokens[i + 1] == image_token_id) image_nums += 1;
                if (tokens[i + 1] == video_token_id) video_nums += 1;
            }
        }
    }

    const appendText = struct {
        fn f(rs: *[3]std.ArrayList(i32), a: std.mem.Allocator, st_idx: i32, text_len: usize) !void {
            var k: usize = 0;
            while (k < text_len) : (k += 1) {
                const p = st_idx + @as(i32, @intCast(k));
                inline for (0..3) |axis| try rs[axis].append(a, p);
            }
        }
    }.f;

    var remain_images = image_nums;
    var remain_videos = video_nums;
    var blk: usize = 0;
    while (blk < image_nums + video_nums) : (blk += 1) {
        // First occurrence at/after st of each remaining modality's pad token;
        // a modality with nothing left (or nothing found) sentinels past `seq`
        // so it is never picked by the `<` comparison below.
        var ed_image: usize = seq + 1;
        if (remain_images > 0) {
            var k = st;
            while (k < seq and tokens[k] != image_token_id) : (k += 1) {}
            if (k < seq) ed_image = k;
        }
        var ed_video: usize = seq + 1;
        if (remain_videos > 0) {
            var k = st;
            while (k < seq and tokens[k] != video_token_id) : (k += 1) {}
            if (k < seq) ed_video = k;
        }
        const use_image = ed_image < ed_video;
        const ed = if (use_image) ed_image else ed_video;
        if (ed > seq) return error.MalformedVisionSequence;

        const g = if (use_image) g: {
            if (image_index >= images.len) return error.MissingImageGrid;
            const gg = images[image_index];
            image_index += 1;
            remain_images -= 1;
            break :g gg;
        } else g: {
            if (video_index >= videos.len) return error.MissingVideoGrid;
            const gg = videos[video_index];
            video_index += 1;
            remain_videos -= 1;
            break :g gg;
        };

        const llm_t: usize = g.t;
        const llm_h: usize = g.h / merge;
        const llm_w: usize = g.w / merge;

        const text_len = ed - st;
        const st_idx: i32 = last_max + 1;
        try appendText(&rows, allocator, st_idx, text_len);
        if (text_len > 0) last_max = st_idx + @as(i32, @intCast(text_len)) - 1;

        // Vision block: t = base+frame, h = base+row, w = base+col.
        const base: i32 = @as(i32, @intCast(text_len)) + st_idx;
        var ti: usize = 0;
        while (ti < llm_t) : (ti += 1) {
            var hi: usize = 0;
            while (hi < llm_h) : (hi += 1) {
                var wi: usize = 0;
                while (wi < llm_w) : (wi += 1) {
                    try rows[0].append(allocator, base + @as(i32, @intCast(ti)));
                    try rows[1].append(allocator, base + @as(i32, @intCast(hi)));
                    try rows[2].append(allocator, base + @as(i32, @intCast(wi)));
                }
            }
        }
        const span = @max(@max(llm_t, llm_h), llm_w);
        const img_max = base + @as(i32, @intCast(span)) - 1;
        if (img_max > last_max) last_max = img_max;

        st = ed + llm_t * llm_h * llm_w;
    }

    // Trailing text.
    if (st < seq) {
        const st_idx: i32 = last_max + 1;
        const text_len = seq - st;
        try appendText(&rows, allocator, st_idx, text_len);
        last_max = st_idx + @as(i32, @intCast(text_len)) - 1;
    }

    std.debug.assert(rows[0].items.len == seq);
    const delta: i32 = (last_max + 1) - @as(i32, @intCast(seq));
    return RopeIndex{
        .pos = .{
            try rows[0].toOwnedSlice(allocator),
            try rows[1].toOwnedSlice(allocator),
            try rows[2].toOwnedSlice(allocator),
        },
        .delta = delta,
        .allocator = allocator,
    };
}

/// Rotary inverse frequencies: `theta^(-2j/rotary_dim)` for j in 0..rotary_dim/2.
/// `rotary_dim = round(head_dim * partial_rotary_factor)`.
pub fn computeInvFreq(out: []f64, rotary_dim: usize, theta: f64) void {
    std.debug.assert(out.len == rotary_dim / 2);
    for (out, 0..) |*o, j| {
        const exp = -@as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rotary_dim));
        o.* = std.math.pow(f64, theta, exp);
    }
}

/// YaRN rope scaling (Peng et al., github.com/jquesnelle/yarn) as HF's
/// `_compute_yarn_parameters` computes it from a `rope_parameters` block with
/// `rope_type: "yarn"` — the mechanism vLLM exposes through `--hf-overrides`
/// to stretch Qwen3.5 past its trained window. Pure f64 host math (no MLX), so
/// it unit-tests straight against the reference values.
///
/// The trunk ropes every one of its rotary tables (attention, the QSA indexer,
/// the MTP head) from ONE of these specs, so the whole model agrees on what a
/// position means (`Yarn.invFreq` → `fillCosSin` / the `mlx_fast_rope` freqs).
pub const Yarn = struct {
    /// `rope_theta` — the base the unscaled frequencies come from.
    theta: f64,
    /// `factor` — how far the window is stretched (262144 × 4 = 1M).
    factor: f64,
    /// `original_max_position_embeddings` — the PRE-TRAINED window. The ramp
    /// boundaries are derived from this, never from the extended one.
    orig_max: u32,
    /// `int(head_dim * partial_rotary_factor)` — the rotating slice of a head.
    rotary_dim: u32,
    /// Wavelengths completing this many rotations across `orig_max` mark the
    /// ends of the ramp: `beta_fast` the fully-extrapolated side, `beta_slow`
    /// the fully-interpolated one. HF defaults: 32 / 1.
    beta_fast: f64 = 32.0,
    beta_slow: f64 = 1.0,
    /// HF's `truncate`: floor/ceil the ramp bounds to whole dims (default true).
    truncate: bool = true,

    pub const CorrectionRange = struct { low: f64, high: f64 };

    /// HF `find_correction_range` — the dim indices the linear ramp spans,
    /// clamped into `[0, rotary_dim - 1]`.
    pub fn correctionRange(self: Yarn) CorrectionRange {
        const dim: f64 = @floatFromInt(self.rotary_dim);
        const mp: f64 = @floatFromInt(self.orig_max);
        const two_log_theta = 2.0 * @log(self.theta);
        // find_correction_dim(r) = dim·ln(orig_max / (2πr)) / (2·ln theta): the
        // freq index whose wavelength completes `r` rotations across the window.
        const corr = struct {
            fn f(rot: f64, d: f64, max_pos: f64, two_log_base: f64) f64 {
                return (d * @log(max_pos / (rot * 2.0 * std.math.pi))) / two_log_base;
            }
        }.f;
        var low = corr(self.beta_fast, dim, mp, two_log_theta);
        var high = corr(self.beta_slow, dim, mp, two_log_theta);
        if (self.truncate) {
            low = @floor(low);
            high = @ceil(high);
        }
        return .{ .low = @max(low, 0.0), .high = @min(high, dim - 1.0) };
    }

    /// The YaRN-corrected inverse frequencies into `out` (`out.len ==
    /// rotary_dim/2`): below `low` the frequency is untouched (extrapolation —
    /// those dims still encode absolute position across the extended window),
    /// at/above `high` it is divided by `factor` (interpolation — a position
    /// `p` now reads as `p/factor`, which is what makes 1M look like 262k), and
    /// the range between blends the two linearly. `factor` 1.0 collapses every
    /// band onto `computeInvFreq`.
    pub fn invFreq(self: Yarn, out: []f64) void {
        std.debug.assert(out.len * 2 == self.rotary_dim);
        const dim: f64 = @floatFromInt(self.rotary_dim);
        const r = self.correctionRange();
        var ramp_denom = r.high - r.low;
        if (ramp_denom == 0) ramp_denom = 0.001; // HF's singularity guard
        for (0..out.len) |i| {
            // pos_freqs[i] = theta^(2i/rotary_dim)
            const pos_freq = std.math.pow(f64, self.theta, @as(f64, @floatFromInt(2 * i)) / dim);
            const inv_extrapolation = 1.0 / pos_freq;
            const inv_interpolation = 1.0 / (self.factor * pos_freq);
            var ramp = (@as(f64, @floatFromInt(i)) - r.low) / ramp_denom;
            ramp = @min(@max(ramp, 0.0), 1.0);
            const extrapolation_factor = 1.0 - ramp;
            out[i] = inv_interpolation * (1.0 - extrapolation_factor) +
                inv_extrapolation * extrapolation_factor;
        }
    }

    /// HF's default `attention_factor` for YaRN (the paper's mscale) — what
    /// `cos`/`sin` get multiplied by when the config does not pin one.
    pub fn attentionFactor(factor: f64) f64 {
        if (factor <= 1.0) return 1.0;
        return 0.1 * @log(factor) + 1.0;
    }

    /// The window this spec actually covers: HF/vLLM both derive
    /// `max_model_len` as `original_max_position_embeddings × factor`, and a
    /// position past it aliases onto one already inside the ramp.
    pub fn contextLen(self: Yarn) u64 {
        const orig: f64 = @floatFromInt(self.orig_max);
        const scaled = @floor(orig * self.factor);
        if (scaled < 0) return 0;
        return @intFromFloat(scaled);
    }
};

/// Fill NeoX-layout cos/sin rows (`[n, rope_dims]`, both halves tiled) for the
/// `n` absolute positions `start + stride*i`: the 3-D table inside the prompt,
/// `abs + delta` past it. `stride 1` is a prefill chunk; the qwen4 QSA indexer
/// ropes its pooled block keys at block-START positions (`stride = ratio`).
///
/// `mscale` is the YaRN attention factor: HF multiplies the cos/sin tables by
/// it, which (both halves of every rotary pair sharing one frequency) scales
/// the rotated slice of q and k by `mscale` — i.e. the attention logits by
/// `mscale²`, exactly the reference's `cos, sin = cos * mscale, sin * mscale`.
/// Pass 1.0 for an unscaled rope.
pub fn fillCosSin(cos: []f32, sin: []f32, positions: PositionContext, start: usize, stride: usize, n: usize, inv_freq: []const f64, sel: []const u8, rope_dims: usize, mscale: f64) void {
    const half = rope_dims / 2;
    std.debug.assert(inv_freq.len == half and sel.len == half);
    std.debug.assert(cos.len == n * rope_dims and sin.len == n * rope_dims);
    for (0..n) |i| {
        const p = start + stride * i;
        const o = i * rope_dims;
        for (0..half) |d| {
            const pid: f64 = @floatFromInt(positions.axisPosition(sel[d], p));
            const angle = pid * inv_freq[d];
            const c: f32 = @floatCast(@cos(angle) * mscale);
            const sn: f32 = @floatCast(@sin(angle) * mscale);
            cos[o + d] = c;
            cos[o + half + d] = c;
            sin[o + d] = sn;
            sin[o + half + d] = sn;
        }
    }
}

test "mrope fillCosSin strided rows == every stride-th row of the contiguous fill" {
    // 3-D table over 12 prompt positions (an image at 4..7 with h/w grids),
    // then decode positions past the table at abs + delta.
    const total: usize = 12;
    var pos: [3 * total]i32 = undefined;
    for (0..total) |i| {
        const t: usize = if (i >= 4 and i < 8) 4 else if (i >= 8) i - 2 else i;
        const h: usize = if (i >= 4 and i < 8) 4 + (i - 4) / 2 else t;
        const w: usize = if (i >= 4 and i < 8) 4 + (i - 4) % 2 else t;
        pos[i] = @intCast(t);
        pos[total + i] = @intCast(h);
        pos[2 * total + i] = @intCast(w);
    }
    const ctx = PositionContext{ .pos = &pos, .total = total, .delta = -2 };
    const rope_dims: usize = 8;
    var inv_freq: [4]f64 = undefined;
    computeInvFreq(&inv_freq, rope_dims, 100.0);
    var sel: [4]u8 = undefined;
    interleavedSelector(&sel, .{ 2, 1, 1 });
    const n_all: usize = 20; // 8 positions past the table
    var cos_all: [n_all * rope_dims]f32 = undefined;
    var sin_all: [n_all * rope_dims]f32 = undefined;
    fillCosSin(&cos_all, &sin_all, ctx, 0, 1, n_all, &inv_freq, &sel, rope_dims, 1.0);
    const stride: usize = 4;
    const n_s: usize = n_all / stride;
    var cos_s: [n_s * rope_dims]f32 = undefined;
    var sin_s: [n_s * rope_dims]f32 = undefined;
    fillCosSin(&cos_s, &sin_s, ctx, 0, stride, n_s, &inv_freq, &sel, rope_dims, 1.0);
    for (0..n_s) |b| {
        for (0..rope_dims) |d| {
            try std.testing.expectEqual(cos_all[b * stride * rope_dims + d], cos_s[b * rope_dims + d]);
            try std.testing.expectEqual(sin_all[b * stride * rope_dims + d], sin_s[b * rope_dims + d]);
        }
    }
    // The image rows are 3-D (h ≠ w angle on the h/w frequencies) and the
    // past-table rows follow abs + delta: row 12 == the angle of position 10.
    try std.testing.expect(cos_all[5 * rope_dims + 1] != cos_all[6 * rope_dims + 1]);
    var one_cos: [rope_dims]f32 = undefined;
    var one_sin: [rope_dims]f32 = undefined;
    const plain = PositionContext{ .pos = &pos, .total = 0, .delta = 0 }; // no table: scalar positions
    fillCosSin(&one_cos, &one_sin, plain, 10, 1, 1, &inv_freq, &sel, rope_dims, 1.0);
    for (0..rope_dims) |d| try std.testing.expectEqual(one_cos[d], cos_all[12 * rope_dims + d]);
}

test "mrope interleaved selector matches reference [11,11,10] over 32 freqs" {
    var sel: [32]u8 = undefined;
    interleavedSelector(&sel, .{ 11, 11, 10 });
    // t (0): 0,3,6,...,30 ; h (1): 1,4,...,31 ; w (2): 2,5,...,29.
    for (0..32) |j| {
        const expect: u8 = switch (j % 3) {
            0 => 0,
            1 => 1,
            2 => if (j <= 29) 2 else 0, // w bounded by 10*3=30 → last is idx 29
            else => unreachable,
        };
        try std.testing.expectEqual(expect, sel[j]);
    }
    // Section sums must equal freq count.
    try std.testing.expectEqual(@as(u32, 32), 11 + 11 + 10);
}

test "mrope get_rope_index single image text+image+text" {
    const a = std.testing.allocator;
    // [A,B,C, vision_start, pad,pad,pad,pad, vision_end, D,E] with grid [1,4,4],
    // merge 2 → 2x2 = 4 image tokens. Hand-derived from get_rope_index.
    const IMG = 248056;
    const VS = 248053;
    const tokens = [_]u32{ 10, 11, 12, VS, IMG, IMG, IMG, IMG, 248054, 20, 21 };
    const images = [_]ImageGrid{.{ .t = 1, .h = 4, .w = 4 }};
    var ri = try getRopeIndex(a, &tokens, &images, &.{}, IMG, 248057, VS, 2);
    defer ri.deinit();

    const exp_t = [_]i32{ 0, 1, 2, 3, 4, 4, 4, 4, 6, 7, 8 };
    const exp_h = [_]i32{ 0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8 };
    const exp_w = [_]i32{ 0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8 };
    try std.testing.expectEqualSlices(i32, &exp_t, ri.pos[0]);
    try std.testing.expectEqualSlices(i32, &exp_h, ri.pos[1]);
    try std.testing.expectEqualSlices(i32, &exp_w, ri.pos[2]);
    try std.testing.expectEqual(@as(i32, -2), ri.delta); // 8+1-11
}

test "mrope get_rope_index pure text is sequential on all axes" {
    const a = std.testing.allocator;
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    var ri = try getRopeIndex(a, &tokens, &.{}, &.{}, 248056, 248057, 248053, 2);
    defer ri.deinit();
    inline for (0..3) |axis| {
        for (ri.pos[axis], 0..) |p, i| try std.testing.expectEqual(@as(i32, @intCast(i)), p);
    }
    try std.testing.expectEqual(@as(i32, 0), ri.delta); // 4+1-5
}

test "mrope get_rope_index video-only exercises the t>1 temporal loop" {
    const a = std.testing.allocator;
    // [A,B,C, vision_start, pad×8, TEXT, D,E] with video grid [2,4,4], merge 2 →
    // llm_t=2 (t is NOT divided by merge), llm_h=llm_w=2 → 2*2*2=8 pad tokens.
    // Hand-derived from get_rope_index's video branch (t,h,w = video_grid_thw[i]).
    const VID = 248057;
    const VS = 248053;
    const tokens = [_]u32{ 10, 11, 12, VS, VID, VID, VID, VID, VID, VID, VID, VID, 248054, 20, 21 };
    const videos = [_]ImageGrid{.{ .t = 2, .h = 4, .w = 4 }};
    var ri = try getRopeIndex(a, &tokens, &.{}, &videos, 248056, VID, VS, 2);
    defer ri.deinit();

    const exp_t = [_]i32{ 0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 8 };
    const exp_h = [_]i32{ 0, 1, 2, 3, 4, 4, 5, 5, 4, 4, 5, 5, 6, 7, 8 };
    const exp_w = [_]i32{ 0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 8 };
    try std.testing.expectEqualSlices(i32, &exp_t, ri.pos[0]);
    try std.testing.expectEqualSlices(i32, &exp_h, ri.pos[1]);
    try std.testing.expectEqualSlices(i32, &exp_w, ri.pos[2]);
    try std.testing.expectEqual(@as(i32, -6), ri.delta); // 9-15
}

test "mrope get_rope_index interleaved image-then-video picks the first marker" {
    const a = std.testing.allocator;
    // [A,B, VS, img_pad×4, VS, vid_pad×4, END]. Both grids [1,4,4] merge 2 → 4
    // pads each, so the two blocks are shape-identical and the test isolates
    // the interleave control flow (ed_image < ed_video picks image first, then
    // falls through to the video block once remain_images hits 0).
    const IMG = 248056;
    const VID = 248057;
    const VS = 248053;
    const tokens = [_]u32{ 1, 2, VS, IMG, IMG, IMG, IMG, VS, VID, VID, VID, VID, 9 };
    const images = [_]ImageGrid{.{ .t = 1, .h = 4, .w = 4 }};
    const videos = [_]ImageGrid{.{ .t = 1, .h = 4, .w = 4 }};
    var ri = try getRopeIndex(a, &tokens, &images, &videos, IMG, VID, VS, 2);
    defer ri.deinit();

    const exp_t = [_]i32{ 0, 1, 2, 3, 3, 3, 3, 5, 6, 6, 6, 6, 8 };
    const exp_h = [_]i32{ 0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8 };
    const exp_w = [_]i32{ 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8 };
    try std.testing.expectEqualSlices(i32, &exp_t, ri.pos[0]);
    try std.testing.expectEqualSlices(i32, &exp_h, ri.pos[1]);
    try std.testing.expectEqualSlices(i32, &exp_w, ri.pos[2]);
    try std.testing.expectEqual(@as(i32, -4), ri.delta); // 9-13
}

test "mrope computeInvFreq base case" {
    var f: [32]f64 = undefined;
    computeInvFreq(&f, 64, 10_000_000.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), f[0], 1e-12);
    // j=1 → theta^(-2/64) = 10e6^(-1/32)
    const expect1 = std.math.pow(f64, 10_000_000.0, -1.0 / 32.0);
    try std.testing.expectApproxEqAbs(expect1, f[1], 1e-12);
}

test "mrope PositionContext maps suffix caches and generated text" {
    // Three axis-major rows, four prompt positions.
    const pos = [_]i32{
        0, 1, 2, 3,
        0, 1, 7, 8,
        0, 1, 9, 10,
    };
    const ctx = PositionContext{
        .pos = &pos,
        .total = 4,
        .delta = -2,
        .base = 2,
    };

    // Cache-relative 0 is absolute prompt position 2.
    try std.testing.expectEqual(@as(usize, 2), ctx.absolutePosition(0));
    try std.testing.expectEqual(@as(i32, 2), ctx.axisPosition(0, 0));
    try std.testing.expectEqual(@as(i32, 7), ctx.axisPosition(1, 0));
    try std.testing.expectEqual(@as(i32, 9), ctx.axisPosition(2, 0));

    // Cache-relative 2 is absolute position 4, just past the table. Generated
    // text collapses to one scalar position on every axis: 4 + (-2) = 2.
    inline for (0..3) |axis| {
        try std.testing.expectEqual(@as(i32, 2), ctx.axisPosition(axis, 2));
    }
}

// ── YaRN rope scaling (the qwen4_exp 262144 → 1048576 extension) ──

/// The shipped Qwen3.8-Flash-Next rope geometry, from the checkpoint's own
/// config.json: head_dim 256 × partial_rotary_factor 0.25 → rotary_dim 64 (so
/// 32 frequencies, which is what `mrope_section` [11,11,10] sums to), theta 1e7,
/// pre-trained window 262144.
const QWEN4_YARN = Yarn{
    .theta = 10_000_000.0,
    .factor = 4.0,
    .orig_max = 262_144,
    .rotary_dim = 64,
};

/// `_compute_yarn_parameters` output for QWEN4_YARN, produced by the reference
/// (tests/dump_qwen4_yarn_fixtures.py, transformers 5.9) — and independently by
/// a transcription of vLLM's `YaRNScalingRotaryEmbedding._compute_inv_freq`,
/// which agrees with HF bit-for-bit here. f64: the references compute in f32,
/// which is 1e-7 sloppier than these.
const QWEN4_YARN_GOLDEN = [_]f64{
    1.0, // 0   extrapolated band (i < low): unscaled
    0.6042963902381328, // 1
    0.36517412725483767, // 2
    0.220673406908459, // 3
    0.1333521432163324, // 4
    0.08058421877614819, // 5
    0.04869675251658631, // 6
    0.029427271762092817, // 7
    0.01778279410038923, // 8
    0.010746078283213174, // 9
    0.006493816315762114, // 10
    0.003924189758484536, // 11
    0.002371373705661655, // 12
    0.0014330125702369627, // 13
    0.0008659643233600654, // 14  last ramp-free extrapolation
    0.0004742398226801046, // 15  blended: 0.90625 of unscaled
    0.00025693505988868084, // 16
    0.0001373497450760004, // 17
    0.00007217387404309113, // 18
    0.0000370722498206804, // 19
    0.000018449222025000475, // 20
    0.000008759770071179004, // 21  blended: 0.34375 of unscaled
    0.0000038498163151487305, // 22 interpolated band (i >= high): exactly /4
    0.0000023264301023242476, // 23
    0.0000014058533129758728, // 24
    0.0000008495520822356398, // 25
    0.0000005133812566142865, // 26
    0.0000003102344401879299, // 27
    0.00000018747355233311396, // 28
    0.00000011328959094002045, // 29
    0.00000006846049085660903, // 30
    0.00000004137042749857954, // 31
};

test "yarn correction range for the qwen4_exp geometry" {
    const r = QWEN4_YARN.correctionRange();
    // low = floor(64·ln(262144/(32·2π))/(2·ln 1e7)) = floor(14.24) = 14
    try std.testing.expectEqual(@as(f64, 14.0), r.low);
    // high = ceil(64·ln(262144/(1·2π))/(2·ln 1e7)) = ceil(21.12) = 22
    try std.testing.expectEqual(@as(f64, 22.0), r.high);
    // Both sit inside the 32-frequency table, so the ramp is a real blend and
    // neither band is empty: 15 dims extrapolated, 7 blended, 10 interpolated.
    try std.testing.expect(r.low < @as(f64, 16));
    try std.testing.expect(r.high < @as(f64, 32));
    try std.testing.expectEqual(@as(u64, 1_048_576), QWEN4_YARN.contextLen());
}

test "yarn inv_freq matches HF _compute_yarn_parameters (qwen4_exp 262k -> 1M)" {
    var f: [32]f64 = undefined;
    QWEN4_YARN.invFreq(&f);
    for (QWEN4_YARN_GOLDEN, 0..) |want, i| {
        try std.testing.expectApproxEqRel(want, f[i], 1e-12);
    }
    // HF's default mscale for factor 4: 0.1·ln 4 + 1.
    try std.testing.expectApproxEqRel(@as(f64, 1.138629436111989), Yarn.attentionFactor(4.0), 1e-15);
    try std.testing.expectEqual(@as(f64, 1.0), Yarn.attentionFactor(1.0));
    try std.testing.expectEqual(@as(f64, 1.0), Yarn.attentionFactor(0.5));
}

test "yarn leaves the low-frequency band unscaled and divides the top band" {
    var plain: [32]f64 = undefined;
    computeInvFreq(&plain, 64, 10_000_000.0);
    var f: [32]f64 = undefined;
    QWEN4_YARN.invFreq(&f);
    const r = QWEN4_YARN.correctionRange();
    for (0..32) |i| {
        const idx: f64 = @floatFromInt(i);
        if (idx <= r.low) {
            // Extrapolation: untouched, so a slow dim still reads absolute
            // position across the whole 1M window.
            try std.testing.expectApproxEqRel(plain[i], f[i], 1e-12);
        } else if (idx >= r.high) {
            // Interpolation: position p now arrives as p/factor.
            try std.testing.expectApproxEqRel(plain[i] / 4.0, f[i], 1e-12);
        } else {
            // Ramp: the RATIO to the unscaled frequency falls linearly from 1 at
            // `low` to 1/factor at `high` — the blend HF/vLLM call YaRN.
            const t = (idx - r.low) / (r.high - r.low);
            const want_ratio = 1.0 - t + t / 4.0;
            try std.testing.expectApproxEqRel(plain[i] * want_ratio, f[i], 1e-12);
        }
        if (i > 0) try std.testing.expect(f[i] <= f[i - 1]); // strictly decreasing
    }
    // The blend is linear in the RATIO, and the ratios at the band edges are the
    // clamp of the ramp: 1 at `low`, 1/factor at `high`.
    const ratio_low = f[@intFromFloat(r.low)] / plain[@intFromFloat(r.low)];
    const ratio_high = f[@intFromFloat(r.high)] / plain[@intFromFloat(r.high)];
    try std.testing.expectApproxEqRel(@as(f64, 1.0), ratio_low, 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 0.25), ratio_high, 1e-12);
}

test "yarn with factor 1.0 is EXACTLY the unscaled spectrum (no-YaRN regression guard)" {
    // Every shipped qwen4_exp checkpoint before the override lands here: the
    // scaled code path must be bit-identical to the old code, not "close".
    var plain: [32]f64 = undefined;
    computeInvFreq(&plain, 64, 10_000_000.0);
    var f: [32]f64 = undefined;
    const none: Yarn = .{ .theta = 10_000_000.0, .factor = 1.0, .orig_max = 262_144, .rotary_dim = 64 };
    none.invFreq(&f);
    for (plain, 0..) |p, i| try std.testing.expectEqual(p, f[i]);
    try std.testing.expectEqual(@as(f64, 1.0), Yarn.attentionFactor(1.0));
}

test "yarn reads a 1M position as 262k in the interpolated band" {
    // The extension's whole point, stated as an equality: for the dims past the
    // ramp, angle_yarn(p) == angle_plain(p/factor). p = 1,048,572 divides by 4
    // exactly, so both sides are the same real number.
    var plain: [32]f64 = undefined;
    computeInvFreq(&plain, 64, 10_000_000.0);
    var yarn_f: [32]f64 = undefined;
    QWEN4_YARN.invFreq(&yarn_f);
    const p: usize = 1_048_572;
    const half: usize = 32;
    var sel: [half]u8 = undefined;
    interleavedSelector(&sel, .{ 11, 11, 10 });
    const all_text = PositionContext{ .pos = &.{}, .total = 0, .delta = 0 };

    var cos_y: [half * 2]f32 = undefined;
    var sin_y: [half * 2]f32 = undefined;
    fillCosSin(&cos_y, &sin_y, all_text, p, 1, 1, &yarn_f, &sel, half * 2, 1.0);
    var cos_p: [half * 2]f32 = undefined;
    var sin_p: [half * 2]f32 = undefined;
    fillCosSin(&cos_p, &sin_p, all_text, p / 4, 1, 1, &plain, &sel, half * 2, 1.0);
    for (22..half) |d| {
        // Absolute tolerance: these are cos/sin values in [-1,1], and some sit
        // close enough to zero that a RELATIVE bound is meaningless.
        try std.testing.expectApproxEqAbs(@as(f32, cos_p[d]), cos_y[d], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, sin_p[d]), sin_y[d], 1e-6);
        // Tiled halves carry the same frequency, so the pair agrees.
        try std.testing.expectEqual(cos_y[d], cos_y[d + half]);
    }
    // And the extrapolated band does NOT collapse: dim 0's frequency is unscaled
    // (exactly `plain[0]`, since ramp = 0 there), so p = 1,048,572 lands on a
    // genuinely different angle than the 262,143 it is "compressed" onto — the
    // model keeps a real sense of absolute distance.
    try std.testing.expectApproxEqRel(@as(f64, 1.0), yarn_f[0] / plain[0], 1e-12);
    try std.testing.expect(@abs(cos_y[0] - cos_p[0]) > 0.05);
}

test "fillCosSin mscale scales the rotated rows only" {
    var f: [32]f64 = undefined;
    QWEN4_YARN.invFreq(&f);
    var sel: [32]u8 = undefined;
    interleavedSelector(&sel, .{ 11, 11, 10 });
    const ctx = PositionContext{ .pos = &.{}, .total = 0, .delta = 0 };
    const rd: usize = 64;
    var cos1: [rd]f32 = undefined;
    var sin1: [rd]f32 = undefined;
    var cos2: [rd]f32 = undefined;
    var sin2: [rd]f32 = undefined;
    fillCosSin(&cos1, &sin1, ctx, 1234, 1, 1, &f, &sel, rd, 1.0);
    // 2.0 is a power of two, so the scaled rows are an exact doubling — this is
    // the "folding the mscale into cos/sin == scaling the rotated q/k" property
    // the fused hd-256 path relies on.
    fillCosSin(&cos2, &sin2, ctx, 1234, 1, 1, &f, &sel, rd, 2.0);
    for (0..rd) |i| {
        try std.testing.expectEqual(@as(f32, cos1[i]) * 2.0, cos2[i]);
        try std.testing.expectEqual(@as(f32, sin1[i]) * 2.0, sin2[i]);
    }
    // The qwen4 window the extension targets: every row is finite and bounded by
    // the mscale (|cos|,|sin| <= 1 before it). No NaN at 1,048,575 positions out.
    const ms = Yarn.attentionFactor(4.0);
    var cos3: [rd]f32 = undefined;
    var sin3: [rd]f32 = undefined;
    fillCosSin(&cos3, &sin3, ctx, 1_048_575, 1, 1, &f, &sel, rd, ms);
    for (0..rd) |i| {
        try std.testing.expect(!std.math.isNan(cos3[i]));
        try std.testing.expect(!std.math.isNan(sin3[i]));
        const msf: f32 = @floatCast(ms);
        const lim = msf + 0.00001;
        try std.testing.expect(@abs(cos3[i]) <= lim);
        try std.testing.expect(@abs(sin3[i]) <= lim);
    }
}

test "YaRN mscale on a 128-wide indexer with 64 rotary dims CAN change top-k" {
    // The skip comment in qsaMaskFromQk claimed mscale cannot change a top-k
    // because it "multiplies every relu(q·k) by one positive constant". That
    // is true of a FULL-head scale. The indexer is 128-wide and only 64 dims
    // rotate, so score = ms²·A + B. Two synthetic blocks whose unscaled
    // winner is the B-heavy one flip under the factor-4 mscale.
    const ms = Yarn.attentionFactor(4.0);
    const ms2 = ms * ms;
    // Block 0: rotary-heavy. Block 1: pass-through-heavy.
    const a0: f64 = 2.0;
    const b0: f64 = 0.0;
    const a1: f64 = 0.0;
    const b1: f64 = 2.2;
    const un0 = a0 + b0;
    const un1 = a1 + b1;
    const sc0 = ms2 * a0 + b0;
    const sc1 = ms2 * a1 + b1;
    try std.testing.expect(un1 > un0); // unscaled winner is block 1
    try std.testing.expect(sc0 > sc1); // scaled winner is block 0
}

fn fixtureF64(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => unreachable,
    };
}

test "qwen4 yarn parity vs HF and vLLM (QWEN4_YARN_FIXTURES)" {
    // Env-gated oracle for the context extension. The frequencies, the mscale and
    // the per-position cos/sin rows come from the REFERENCE —
    // `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["yarn"]`, cross-checked
    // inside the dumper against a transcription of vLLM's
    // `YaRNScalingRotaryEmbedding` — for THIS checkpoint's own geometry, dumped by
    // `tests/dump_qwen4_yarn_fixtures.py`. Dormant in normal CI:
    //
    //     QWEN4_YARN_FIXTURES=/tmp/qwen4_yarn.json \
    //         zig build test -Dtest-filter="qwen4 yarn parity"
    const path_z = std.c.getenv("QWEN4_YARN_FIXTURES") orelse return error.SkipZigTest;
    const path = std.mem.span(path_z);
    if (path.len == 0) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const data = try reader_state.interface.allocRemaining(std.testing.allocator, .limited(8 << 20));
    defer std.testing.allocator.free(data);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, data, .{});
    defer parsed.deinit();
    const fx = parsed.value.object;

    const rotary_dim: u32 = @intCast(fx.get("rotary_dim").?.integer);
    const spec = Yarn{
        .theta = fixtureF64(fx.get("rope_theta").?),
        .factor = fixtureF64(fx.get("factor").?),
        .orig_max = @intCast(fx.get("original_max_position_embeddings").?.integer),
        .rotary_dim = rotary_dim,
        .beta_fast = fixtureF64(fx.get("beta_fast").?),
        .beta_slow = fixtureF64(fx.get("beta_slow").?),
        .truncate = fx.get("truncate").?.bool,
    };
    // The dumper only describes a qwen4_exp-shaped run; anything else means the
    // fixture was regenerated against a different checkpoint.
    std.debug.assert(rotary_dim == 64);
    try std.testing.expectEqual(@as(u32, 256), @as(u32, @intCast(fx.get("head_dim").?.integer)));

    // 1. The ramp bounds.
    const cr = spec.correctionRange();
    try std.testing.expectApproxEqAbs(fixtureF64(fx.get("correction_low").?), cr.low, 1e-12);
    try std.testing.expectApproxEqAbs(fixtureF64(fx.get("correction_high").?), cr.high, 1e-12);

    // 2. The frequencies — against BOTH references. vLLM's transcription is
    // f64, so this engine's f64 table has to match it in f64; HF computes in
    // f32, which is why its column is checked at f32 width.
    const want_vllm = fx.get("vllm_inv_freq").?.array;
    const want_hf = fx.get("inv_freq").?.array;
    try std.testing.expectEqual(@as(usize, rotary_dim / 2), want_vllm.items.len);
    var got: [32]f64 = undefined;
    spec.invFreq(&got);
    var max_vllm: f64 = 0;
    var max_hf: f64 = 0;
    for (want_vllm.items, want_hf.items, 0..) |wv, wh, i| {
        max_vllm = @max(max_vllm, @abs(got[i] - fixtureF64(wv)) / fixtureF64(wv));
        max_hf = @max(max_hf, @abs(got[i] - fixtureF64(wh)) / fixtureF64(wh));
    }
    try std.testing.expect(max_vllm < 1e-14);
    try std.testing.expect(max_hf < 1e-6); // HF's own float32 rounding

    // 3. The mscale.
    try std.testing.expectApproxEqAbs(
        fixtureF64(fx.get("attention_factor").?),
        Yarn.attentionFactor(spec.factor),
        1e-12,
    );

    // 4. The actual tables, at positions spanning the pre-trained window, its
    // edge, and the extended range it could not previously address. The engine
    // fills cos/sin in f64 and rounds once to f32; the reference rounds the
    // same angles, so the rows agree to one f32 step.
    const sel_half = rotary_dim / 2;
    var sel: [32]u8 = undefined;
    var section = [3]u32{ 0, 0, 0 };
    for (fx.get("mrope_section").?.array.items, 0..) |item, i| section[i] = @intCast(item.integer);
    interleavedSelector(sel[0..sel_half], section);
    const text_only = PositionContext{ .pos = &.{}, .total = 0, .delta = 0 };
    const ms = Yarn.attentionFactor(spec.factor);

    var cos: [64]f32 = undefined;
    var sin: [64]f32 = undefined;
    var rows_seen: usize = 0;
    var it = fx.get("rows").?.object.iterator();
    while (it.next()) |entry| {
        const p = try std.fmt.parseInt(usize, entry.key_ptr.*, 10);
        rows_seen += 1;
        const row = entry.value_ptr.*.object;
        // Unscaled: the plain angles at this position.
        fillCosSin(&cos, &sin, text_only, p, 1, 1, &got, sel[0..], rotary_dim, 1.0);
        for (row.get("cos").?.array.items, 0..) |w, d| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(fixtureF64(w))), cos[d], 1e-6);
        }
        for (row.get("sin").?.array.items, 0..) |w, d| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(fixtureF64(w))), sin[d], 1e-6);
        }
        // Scaled: the same rows with the mscale folded in — what the engine
        // hands to `applyMrope`, and what vLLM bakes into its cos_sin_cache.
        fillCosSin(&cos, &sin, text_only, p, 1, 1, &got, sel[0..], rotary_dim, ms);
        for (row.get("cos_scaled").?.array.items, 0..) |w, d| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(fixtureF64(w))), cos[d], 1e-6);
        }
        for (row.get("sin_scaled").?.array.items, 0..) |w, d| {
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(fixtureF64(w))), sin[d], 1e-6);
        }
    }
    try std.testing.expect(rows_seen >= 12);
    std.debug.print("[qwen4-yarn] {d} freqs + {d} position rows (through {d}) match HF/vLLM\n", .{
        got.len, rows_seen, fx.get("extended_max_position_embeddings").?.integer,
    });
}
