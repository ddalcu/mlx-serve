//! Image prompt rows and their identities. ds4 compares fingerprints as well as token ids when reusing KV.
const std = @import("std");
const abi = @import("../../abi/engine.zig");
const png = @import("../../sdk/media/png.zig");
const meta = @import("../../sdk/quant/gguf/engine_pick.zig");

pub const Tokens = struct {
    start: i32,
    pad: i32,
    end: i32,
    placeholder: []const u8,
};

/// These are wire-format marker names, never vocabulary ids or checkpoint-name guesses.
pub fn tokens(io: std.Io, a: std.mem.Allocator, path: []const u8) ?Tokens {
    inline for (.{
        [3][]const u8{ "<|vision_start|>", "<|image_pad|>", "<|vision_end|>" },
        [3][]const u8{ "<|begin_of_image|>", "<|image|>", "<|end_of_image|>" },
    }) |names| {
        const ids = meta.readTokenIds(io, a, path, 3, names) catch return null;
        if (ids[0] != null and ids[1] != null and ids[2] != null) return .{
            .start = @intCast(ids[0].?),
            .pad = @intCast(ids[1].?),
            .end = @intCast(ids[2].?),
            .placeholder = names[0] ++ names[1] ++ names[2],
        };
    }
    return null;
}

pub fn Cache(comptime B: type, comptime a: std.mem.Allocator) type {
    return struct {
        const Self = @This();
        const Key = struct { hash: u64, width: u32, height: u32 };
        images: std.AutoHashMapUnmanaged(Key, B.Embedding) = .empty,
        /// The last successful sync's image identities, used to authenticate a client continuation.
        spans: []B.Span = &.{},

        pub fn deinit(self: *Self) void {
            var it = self.images.valueIterator();
            while (it.next()) |emb| B.Ds4Engine.freeEmbedding(emb);
            self.images.deinit(a);
            a.free(self.spans);
        }

        fn embedding(self: *Self, engine: *B.Ds4Engine, m: abi.Media) !B.Embedding {
            if (m.kind != .image) return error.ImagesOnly;
            const pixels = abi.items(u8, m.data);
            const size = std.math.mul(usize, m.width, m.height) catch return error.InvalidImage;
            if (m.width == 0 or m.height == 0 or size > std.math.maxInt(usize) / 3 or pixels.len != size * 3) return error.InvalidImage;
            const key: Key = .{ .hash = std.hash.Wyhash.hash(0, pixels), .width = m.width, .height = m.height };
            if (self.images.get(key)) |emb| return emb;
            const encoded = try png.encodeRgb(a, pixels, m.width, m.height);
            defer a.free(encoded);
            var emb = try engine.encodeImage(encoded);
            errdefer B.Ds4Engine.freeEmbedding(&emb);
            if (emb.data == null or emb.token_count == 0) return error.VisionEncodeFailed;
            try self.images.put(a, key, emb);
            return emb;
        }

        /// Expands each placeholder to the tower's actual row count; pixel count alone is not the row count.
        pub fn expand(self: *Self, engine: *B.Ds4Engine, v: Tokens, prompt: []const i32, media: []const abi.Media) ![]i32 {
            var out: std.ArrayList(i32) = .empty;
            errdefer out.deinit(a);
            var k: usize = 0;
            var i: usize = 0;
            while (i < prompt.len) {
                if (i + 2 < prompt.len and prompt[i] == v.start and prompt[i + 1] == v.pad and prompt[i + 2] == v.end) {
                    if (k >= media.len) return error.MediaPlaceholderMismatch;
                    const emb = try self.embedding(engine, media[k]);
                    k += 1;
                    try out.append(a, v.start);
                    try out.appendNTimes(a, v.pad, emb.token_count);
                    try out.append(a, v.end);
                    i += 3;
                } else {
                    try out.append(a, prompt[i]);
                    i += 1;
                }
            }
            if (k != media.len) return error.MediaPlaceholderMismatch;
            return out.toOwnedSlice(a);
        }

        fn imageSpans(self: *Self, engine: *B.Ds4Engine, v: Tokens, prompt: []const i32, media: []const abi.Media) ![]B.Span {
            var spans: std.ArrayList(B.Span) = .empty;
            errdefer spans.deinit(a);
            var i: usize = 0;
            while (i + 1 < prompt.len) : (i += 1) {
                if (prompt[i] != v.start or prompt[i + 1] != v.pad) continue;
                var j = i + 1;
                while (j < prompt.len and prompt[j] == v.pad) j += 1;
                const k = spans.items.len;
                if (j == prompt.len or prompt[j] != v.end or k >= media.len) return error.MediaPlaceholderMismatch;
                const emb = try self.embedding(engine, media[k]);
                if (emb.token_count != j - (i + 1)) return error.MediaPlaceholderMismatch;
                try spans.append(a, .{ .token_start = @intCast(i + 1), .embedding = emb });
                i = j;
            }
            if (spans.items.len != media.len) return error.MediaPlaceholderMismatch;
            return spans.toOwnedSlice(a);
        }

        /// Text-identical image markers cannot authorize a reply splice after the underlying pixels changed.
        pub fn continues(self: *Self, engine: *B.Ds4Engine, media: []const abi.Media) !bool {
            if (media.len < self.spans.len) return false;
            for (self.spans, media[0..self.spans.len]) |old, m| {
                const emb = try self.embedding(engine, m);
                if (emb.token_count != old.embedding.token_count or !std.mem.eql(u8, &emb.fingerprint, &old.embedding.fingerprint)) return false;
            }
            return true;
        }

        pub fn sync(self: *Self, engine: *B.Ds4Engine, session: *B.Ds4Session, v: Tokens, prompt: []const i32, media: []const abi.Media, cancel: *std.atomic.Value(bool)) !void {
            const spans = try self.imageSpans(engine, v, prompt, media);
            errdefer a.free(spans);
            try session.syncMultimodal(prompt, spans, cancel);
            a.free(self.spans);
            self.spans = spans;
            self.prune();
        }

        /// No fixed-count eviction cap: every stale image row allocation is released after a successful sync.
        fn prune(self: *Self) void {
            var i: usize = 0;
            while (i < self.images.capacity()) : (i += 1) {
                // Hash-map removals do not invalidate other slots; restart iteration after a removal.
                var it = self.images.iterator();
                const removed = while (it.next()) |entry| {
                    const used = for (self.spans) |sp| {
                        if (std.mem.eql(u8, &sp.embedding.fingerprint, &entry.value_ptr.fingerprint)) break true;
                    } else false;
                    if (!used) {
                        var emb = entry.value_ptr.*;
                        const key = entry.key_ptr.*;
                        _ = self.images.remove(key);
                        B.Ds4Engine.freeEmbedding(&emb);
                        break true;
                    }
                } else false;
                if (!removed) break;
            }
        }
    };
}
