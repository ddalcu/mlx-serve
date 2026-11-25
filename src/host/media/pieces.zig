//! Request media pieces and their shared pixel/PCM cache keys.

const std = @import("std");
const MediaItem = @import("placeholders.zig").MediaItem;
const scheduler_mod = @import("../gpu_thread/requests.zig");

/// The encoder inputs for a request's items: one piece per tower call (an
/// LFM2-VL tile, else the whole item), keyed by its pixels/PCM; an item's key
/// folds its pieces'.
pub const MediaPieces = struct {
    pieces: []scheduler_mod.VisionItem,
    piece_keys: []u64,
    item_keys: []u64,

    pub fn init(allocator: std.mem.Allocator, items: []const MediaItem) !MediaPieces {
        var pieces = std.ArrayList(scheduler_mod.VisionItem).empty;
        defer pieces.deinit(allocator);
        for (items) |item| switch (item) {
            .image => |imgs| for (imgs) |img| try pieces.append(allocator, .{ .image = .{
                .pixels = img.pixels,
                .width = img.width,
                .height = img.height,
                .grid_h = img.grid_h,
                .grid_w = img.grid_w,
            } }),
            .video => |vid| try pieces.append(allocator, .{ .video = .{
                .pixels = vid.pixels,
                .grid_t = vid.grid_t,
                .grid_h = vid.grid_h,
                .grid_w = vid.grid_w,
                .fps = vid.fps,
            } }),
            .audio => |au| try pieces.append(allocator, .{ .audio = au.samples }),
        };
        const piece_keys = try allocator.alloc(u64, pieces.items.len);
        errdefer allocator.free(piece_keys);
        for (pieces.items, piece_keys) |piece, *k| k.* = pieceKey(piece);
        const item_keys = try allocator.alloc(u64, items.len);
        errdefer allocator.free(item_keys);
        var p: usize = 0;
        for (items, item_keys) |item, *k| {
            const n = piecesOf(item);
            var h = std.hash.Wyhash.init(0x5ec0de);
            h.update(std.mem.sliceAsBytes(piece_keys[p .. p + n]));
            k.* = h.final() | 1;
            p += n;
        }
        return .{ .pieces = try pieces.toOwnedSlice(allocator), .piece_keys = piece_keys, .item_keys = item_keys };
    }

    pub fn deinit(self: *MediaPieces, allocator: std.mem.Allocator) void {
        allocator.free(self.pieces);
        allocator.free(self.piece_keys);
        allocator.free(self.item_keys);
    }
};

pub fn piecesOf(item: MediaItem) usize {
    return if (item == .image) item.image.len else 1;
}

fn pieceKey(piece: scheduler_mod.VisionItem) u64 {
    var h = std.hash.Wyhash.init(@backingInt(std.meta.activeTag(piece)));
    switch (piece) {
        .image => |im| {
            h.update(std.mem.asBytes(&[_]u32{ im.width, im.height, im.grid_h, im.grid_w }));
            h.update(im.pixels);
        },
        .video => |vd| {
            h.update(std.mem.asBytes(&[_]u32{ vd.grid_t, vd.grid_h, vd.grid_w, @bitCast(vd.fps) }));
            h.update(vd.pixels);
        },
        .audio => |pcm| h.update(pcm),
    }
    return h.final();
}
