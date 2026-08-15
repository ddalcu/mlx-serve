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
