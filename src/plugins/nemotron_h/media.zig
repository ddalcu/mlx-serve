//! The RADIO tower's media glue: its placeholder token runs (`radioMediaTokens`, `appendRadioVideoRun`) and the
//! video frame decode.

const std = @import("std");
const chat_mod = @import("../../host/chat/chat.zig");
const model_mod = @import("../../sdk/config/model.zig");
const radio_vision = @import("radio_vision.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;

const MediaTokens = @import("../../host/media/placeholders.zig").MediaTokens;
const decodeFrameUrl = @import("../../host/media/decode.zig").decodeFrameUrl;

/// One RADIO clip: each tubelet is its label, `<img>`, the kept `<image>` ids, `</img>`.
/// The label is tokenized alone so the specials stay atoms. A short last group
/// omits the frame the encoder repeats. Dropped positions are absent, which is
/// the sequence EVS leaves after it prunes the full run.
pub fn appendRadioVideoRun(
    allocator: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    vid: *const chat_mod.VideoData,
    kept: usize,
    mask: ?[]const bool,
    mt: MediaTokens,
    tok: anytype,
) !void {
    const geom = radio_vision.videoGeometry(vid.grid_t, vid.grid_h, vid.grid_w) orelse return error.InvalidVisionInput;
    const full: usize = @as(usize, geom.groups) * geom.spatial;
    const bits = mask orelse return error.InvalidVisionInput;
    if (bits.len != full) return error.InvalidVisionInput;
    if (mt.config.image_token_id == 0 or mt.video_sentinel == 0 or mt.img_end == 0) return error.InvalidVisionInput;
    if (mt.video_sentinel == mt.config.image_token_id) return error.InvalidVisionInput;
    var seen: usize = 0;
    var g: u32 = 0;
    while (g < geom.groups) : (g += 1) {
        var buf: [96]u8 = undefined;
        const label = radio_vision.videoGroupSegment(&buf, g, geom.frames);
        if (label.len == 0) return error.InvalidVisionInput;
        const label_ids = try tok.encode(allocator, label);
        defer allocator.free(label_ids);
        try ids.appendSlice(allocator, label_ids);
        try ids.append(allocator, mt.video_sentinel);
        const base: usize = @as(usize, g) * geom.spatial;
        var i: usize = 0;
        while (i < geom.spatial) : (i += 1) {
            if (!bits[base + i]) continue;
            try ids.append(allocator, mt.config.image_token_id);
            seen += 1;
        }
        try ids.append(allocator, mt.img_end);
    }
    if (seen != kept) return error.InvalidVisionInput;
}

pub fn radioMediaTokens(lm: *LoadedModel, config: *const model_mod.ModelConfig) MediaTokens {
    if (!config.radio_vision) return .{ .config = config };
    const tok = lm.tokenizer orelse return .{ .config = config };
    return .{
        .config = config,
        .video_sentinel = tok.specialTokenId("<img>") orelse 0,
        .img_end = tok.specialTokenId("</img>") orelse 0,
    };
}

/// RADIO video: every frame resized to frame 0's patch grid, CLIP-normalized,
/// concatenated as float32 CHW. `grid_h` / `grid_w` are pixels, `grid_t` is the
/// raw frame count. Null when any frame is not a decodable image.
pub fn decodeRadioVideo(allocator: std.mem.Allocator, frame_urls: []const []const u8) ?chat_mod.VideoData {
    if (frame_urls.len == 0) return null;
    var vb: ?radio_vision.VideoBuilder = null;
    for (frame_urls, 0..) |url, i| {
        const frame = decodeFrameUrl(allocator, url, true) orelse return cancelRadioVideo(allocator, vb);
        defer frame.deinit(allocator);
        if (vb == null) vb = radio_vision.VideoBuilder.init(allocator, frame_urls.len, frame.h, frame.w) catch return null;
        vb.?.add(allocator, i, frame.rgb, frame.h, frame.w) catch return cancelRadioVideo(allocator, vb);
    }
    const prepared = vb.?.finish() catch return cancelRadioVideo(allocator, vb);
    log.info("  Decoded {d} frames → RADIO video {d}x{d} ({d} tubelets x {d} tokens)\n", .{
        frame_urls.len, prepared.w, prepared.h, prepared.groups, prepared.tokens,
    });
    return .{ .pixels = prepared.pixels, .grid_t = prepared.frames, .grid_h = prepared.h, .grid_w = prepared.w };
}

fn cancelRadioVideo(allocator: std.mem.Allocator, vb: ?radio_vision.VideoBuilder) ?chat_mod.VideoData {
    if (vb) |v| allocator.free(v.buf);
    return null;
}
