//! Ling VL's M-RoPE layout: the axis selector and the body-diagonal position table (the
//! checkpoint's `get_rope_index`), one plugin's code the engine reaches through `MropeFacts.selector`
//! (`driver.mropeAngles`) and the host reaches through `mrope_media.zig`.

const std = @import("std");
const mrope_mod = @import("../../sdk/vision/mrope.zig");


/// The bailing/Ling VL layout (`MropeFacts.selector == .bailing`): the LAST `mrope_section[0]` frequency slots take the
/// temporal axis and the leading slots alternate H (even) / W (odd). The checkpoint's `apply_video_rope` chunks
/// the doubled `[freqs | freqs]` table as `[H+W interleaved | T | H+W | T]` — which is exactly this half-slot
/// pattern once the duplicated halves are mirrored (and pairs stay axis-consistent under `rotate_half`/interleaved).
pub fn bailingSelector(sel: []u8, mrope_section: [3]u32) void {
    const freq_dim = sel.len;
    const t: usize = @min(mrope_section[0], freq_dim);
    for (0..freq_dim) |d| sel[d] = if (d + t >= freq_dim) 0 else if (d % 2 == 0) 1 else 2;
}

/// Ling VL's (`bailing_moe_v3_vl`) position table: the checkpoint's `get_rope_index` — SGLang's
/// Bailing 3-D layout with body-diagonal symmetry, NOT Qwen's. Text segments advance all three
/// axes one per token; an image block whose `text_len`-token preamble starts at running position
/// `st_idx` places its merged (T,H,W) grid at `t = st_idx + text_len + 2·t`, `h = t + h −
/// (H−1)/2`, `w = t + w − (W−1)/2` (h/w are CENTRED on the diagonal — they can precede the block's
/// own text), and the next segment resumes at max(t)+1. Images only: video is not served.
pub fn getRopeIndexBailing(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    images: []const mrope_mod.ImageGrid, // full patch grids (t = temporal patches, 1 for a still image)
    image_token_id: u32,
    merge: u32,
) !mrope_mod.RopeIndex {
    const seq = tokens.len;
    var rows: [3]std.ArrayList(i32) = .{ .empty, .empty, .empty };
    errdefer for (&rows) |*r| r.deinit(allocator);

    var st_idx: i32 = 0;
    var st: usize = 0;
    for (images) |g| {
        var ed: usize = seq + 1;
        var i = st;
        while (i < seq) : (i += 1) if (tokens[i] == image_token_id) {
            ed = i;
            break;
        };
        if (ed > seq) return error.MropeImageRunMissing;
        const text_len: i32 = @intCast(ed - st);
        for (0..@intCast(text_len)) |j| {
            const p = st_idx + @as(i32, @intCast(j));
            for (0..3) |ax| try rows[ax].append(allocator, p);
        }
        const t_n = g.t;
        const h_n = g.h / merge;
        const w_n = g.w / merge;
        const base = st_idx + text_len;
        const hmid: i32 = @intCast((h_n - 1) / 2);
        const wmid: i32 = @intCast((w_n - 1) / 2);
        for (0..t_n) |tt| {
            const ti = base + 2 * @as(i32, @intCast(tt));
            for (0..h_n) |hh| for (0..w_n) |ww| {
                try rows[0].append(allocator, ti);
                try rows[1].append(allocator, ti + @as(i32, @intCast(hh)) - hmid);
                try rows[2].append(allocator, ti + @as(i32, @intCast(ww)) - wmid);
            };
        }
        st_idx = base + 2 * @as(i32, @intCast(t_n - 1)) + 1;
        st = ed + t_n * h_n * w_n;
    }
    for (st..seq) |j| {
        const p = st_idx + @as(i32, @intCast(j - st));
        for (0..3) |ax| try rows[ax].append(allocator, p);
    }
    for (0..3) |ax| if (rows[ax].items.len != seq) {
        for (0..3) |q| rows[q].deinit(allocator);
        return error.MropeImageRunMissing;
    };

    var pos: [3][]i32 = undefined;
    for (0..3) |ax| pos[ax] = try rows[ax].toOwnedSlice(allocator);
    var max_t: i32 = std.math.minInt(i32);
    for (pos[0]) |p| max_t = @max(max_t, p);
    return .{ .pos = pos, .delta = max_t + 1 - @as(i32, @intCast(seq)), .allocator = allocator };
}
