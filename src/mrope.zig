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
/// IMAGE inputs. `tokens` is the POST-EXPANSION id sequence (each image's single
/// `<|image_pad|>` already expanded to its merged-grid token count). `images`
/// lists the full patch grid per image in document order. Returns 3×seq position
/// ids (caller owns via `RopeIndex.deinit`) and the decode delta.
///
/// Video tokens are not handled (returns error.VideoUnsupported) — the Qwen
/// vision feature ships image-only in its first cut.
pub fn getRopeIndex(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    images: []const ImageGrid,
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

    // image_nums = number of vision_start tokens immediately followed by image_token.
    var image_nums: usize = 0;
    {
        var i: usize = 0;
        while (i + 1 < seq) : (i += 1) {
            if (tokens[i] == vision_start_token_id) {
                if (tokens[i + 1] == image_token_id) image_nums += 1;
                if (tokens[i + 1] == video_token_id) return error.VideoUnsupported;
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

    var img: usize = 0;
    while (img < image_nums) : (img += 1) {
        // First image-pad at/after st.
        var ed: usize = st;
        while (ed < seq and tokens[ed] != image_token_id) : (ed += 1) {}
        if (ed >= seq) return error.MalformedVisionSequence;
        if (image_index >= images.len) return error.MissingImageGrid;
        const g = images[image_index];
        image_index += 1;

        const llm_t: usize = g.t;
        const llm_h: usize = g.h / merge;
        const llm_w: usize = g.w / merge;

        const text_len = ed - st;
        const st_idx: i32 = last_max + 1;
        try appendText(&rows, allocator, st_idx, text_len);
        if (text_len > 0) last_max = st_idx + @as(i32, @intCast(text_len)) - 1;

        // Image block: t = base+frame, h = base+row, w = base+col.
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
    var ri = try getRopeIndex(a, &tokens, &images, IMG, 248057, VS, 2);
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
    var ri = try getRopeIndex(a, &tokens, &.{}, 248056, 248057, 248053, 2);
    defer ri.deinit();
    inline for (0..3) |axis| {
        for (ri.pos[axis], 0..) |p, i| try std.testing.expectEqual(@as(i32, @intCast(i)), p);
    }
    try std.testing.expectEqual(@as(i32, 0), ri.delta); // 4+1-5
}

test "mrope computeInvFreq base case" {
    var f: [32]f64 = undefined;
    computeInvFreq(&f, 64, 10_000_000.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), f[0], 1e-12);
    // j=1 → theta^(-2/64) = 10e6^(-1/32)
    const expect1 = std.math.pow(f64, 10_000_000.0, -1.0 / 32.0);
    try std.testing.expectApproxEqAbs(expect1, f[1], 1e-12);
}
