//! GLM-5.3's decode state: per layer the attention's KV as two bounded grow lanes (`sdk_ext.kv.Lanes(G).Grow`), the
//! normalized latent `[1, rows, kv_lora_rank]` and the roped key `[1, rows, qk_rope_head_dim]`, and on each full
//! indexer layer the indexer's roped key `[1, rows, index_head_dim]`; every lane bf16 as the projections produce it.
//! Every lane holds exactly `cap` positions: the module sets the cap per phase (`resize`: a prompt's positions at its
//! pass, the request's prompt and generation at its decode handover), and a lane is allocated at its first append or
//! moved to a buffer of the new cap with the rows it holds. A later prompt truncates the lanes to the prefix it keeps
//! (`truncateTo`) and appends after it. An append past the cap is refused by name (`BoundedLaneFull`).

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const glm = @import("glm_moe_dsa.zig");

pub fn Cache(comptime G: type) type {
    return struct {
        const Self = @This();
        const L = sdk_ext.kv.Lanes(G);
        pub const T = G.T;

        a: std.mem.Allocator,
        latent: []L.Grow,
        rope: []L.Grow,
        /// The full layers' indexer keys (null on shared layers).
        index: []?L.Grow,
        /// The positions every lane holds.
        len: u32 = 0,
        cap: u32,
        /// One position's bytes in a latent, a rope and an index lane (bf16).
        row_bytes: [3]u64,

        pub fn init(a: std.mem.Allocator, c: *const glm.Config, cap: u32) !Self {
            const n = c.n_layers;
            const latent = try a.alloc(L.Grow, n);
            errdefer a.free(latent);
            const rope = try a.alloc(L.Grow, n);
            errdefer a.free(rope);
            const index = try a.alloc(?L.Grow, n);
            for (latent, rope, index, 0..) |*lt, *rp, *ix, l| {
                lt.* = L.Grow.init(cap, cap);
                rp.* = L.Grow.init(cap, cap);
                ix.* = if (c.isFull(@intCast(l))) L.Grow.init(cap, cap) else null;
            }
            return .{ .a = a, .latent = latent, .rope = rope, .index = index, .cap = cap, .row_bytes = .{ 2 * @as(u64, c.kv_lora_rank), 2 * @as(u64, c.qk_rope_head_dim), 2 * @as(u64, c.index_head_dim) } };
        }

        /// Every empty lane allocated at its bound (bf16 zeros, as its first append would), evaluated: a prompt pass then
        /// appends into lanes it does not allocate, so the allocator's size classes (on for the pass's transients) never
        /// round a lane.
        pub fn preallocate(self: *Self, g: *G, c: *const glm.Config) !void {
            // Only the kept handles outlive this: a first append then writes its lane in place (no other reference).
            const m = g.mark();
            defer g.resetTo(m);
            var news: std.ArrayList(T) = .empty;
            defer news.deinit(self.a);
            for (self.latent, self.rope, self.index) |*lt, *rp, *ix| {
                const dims = [3]u32{ c.kv_lora_rank, c.qk_rope_head_dim, c.index_head_dim };
                const lanes = [3]?*L.Grow{ lt, rp, if (ix.*) |*x| x else null };
                for (lanes, dims) |lane_, d| {
                    const lane = lane_ orelse continue;
                    if (lane.buf != null) continue;
                    const cap = lane.bounded_cap orelse continue;
                    lane.buf = g.keep(try g.zeros(&.{ 1, @intCast(cap), @intCast(d) }, .bfloat16));
                    lane.cap = cap;
                    try news.append(self.a, lane.buf.?);
                }
            }
            if (news.items.len > 0) try g.evalAll(news.items);
        }

        /// Every lane bounded at exactly `cap` positions (at least the positions held), layer by layer: a lane with
        /// rows moves to a `cap`-row buffer, evaluated before the next layer's, so the old and the new buffers of one
        /// layer at most are live at once. No-op at the same cap.
        pub fn resize(self: *Self, g: *G, cap: u32) !void {
            if (cap < self.len) return error.BoundedLaneFull;
            for (self.latent, self.rope, self.index, 0..) |*lt, *rp, *ix, l| {
                const m = g.mark();
                defer g.resetTo(m);
                try lt.resize(g, cap);
                try rp.resize(g, cap);
                if (ix.*) |*x| try x.resize(g, cap);
                var bufs: [3]T = undefined;
                const live = self.buffers(@intCast(l), &bufs);
                if (live.len > 0) try g.evalAll(live);
            }
            self.cap = cap;
        }

        /// The bytes the lanes' buffers hold now, each rounded up to `page` as MLX allocates it.
        pub fn allocatedBytes(self: *const Self, page: u64) u64 {
            var n: u64 = 0;
            for (self.latent, self.rope, self.index) |lt, rp, ix| {
                if (lt.buf != null) n += bufferBytes(@as(u64, lt.cap) * self.row_bytes[0], page);
                if (rp.buf != null) n += bufferBytes(@as(u64, rp.cap) * self.row_bytes[1], page);
                if (ix) |x| if (x.buf != null) {
                    n += bufferBytes(@as(u64, x.cap) * self.row_bytes[2], page);
                };
            }
            return n;
        }

        pub fn deinit(self: *Self, g: *G) void {
            for (self.latent, self.rope, self.index) |*lt, *rp, *ix| {
                lt.deinit(g);
                rp.deinit(g);
                if (ix.*) |*x| x.deinit(g);
            }
            self.a.free(self.latent);
            self.a.free(self.rope);
            self.a.free(self.index);
            self.* = undefined;
        }

        /// Layer `l`'s new rows: the latent `[1, n, kv_lora_rank]`, the roped key `[1, n, rope]` and, on a full layer,
        /// the indexer key `[1, n, index_head_dim]`.
        pub fn append(self: *Self, g: *G, l: u32, latent: T, rope: T, index: ?T) !void {
            try self.latent[l].append(g, latent);
            try self.rope[l].append(g, rope);
            if (index) |x| try (if (self.index[l]) |*ix| ix else return error.IndexLaneOnSharedLayer).append(g, x);
        }

        /// Layer `l`'s lanes as `[1, 1, n, d]` views (n = the positions it holds).
        pub fn latentView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &self.latent[l]);
        }

        pub fn ropeView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &self.rope[l]);
        }

        pub fn indexView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &(self.index[l] orelse return error.IndexLaneOnSharedLayer));
        }

        fn viewOf(g: *G, lane: *const L.Grow) !T {
            const v = (try lane.view(g)) orelse return error.LaneEmpty;
            const s = g.shapeOf(v);
            return g.reshape(v, &.{ 1, 1, s.dim(1), s.dim(2) });
        }

        /// Every lane's buffer (the arrays an eval of the layer's appends covers).
        pub fn buffers(self: *const Self, l: u32, out: *[3]T) []const T {
            var n: usize = 0;
            for ([_]?T{ self.latent[l].buf, self.rope[l].buf, if (self.index[l]) |ix| ix.buf else null }) |b| if (b) |x| {
                out[n] = x;
                n += 1;
            };
            return out[0..n];
        }

        /// After a forward of `n` rows (every layer appended them).
        pub fn commit(self: *Self, n: u32) void {
            self.len += n;
        }

        /// Keep the first `n` positions of every lane (a prefix the next prompt reuses); the capacity stays.
        pub fn truncateTo(self: *Self, n: u32) void {
            for (self.latent, self.rope, self.index) |*lt, *rp, *ix| {
                lt.truncateTo(n);
                rp.truncateTo(n);
                if (ix.*) |*x| x.truncateTo(n);
            }
            self.len = @min(self.len, n);
        }
    };
}

/// The bytes MLX's allocator takes for a request of `bytes` while its size classes are on (the patched allocator's
/// `mlx_set_alloc_size_classes`, which GLM-5.3's module turns on only while the prompt's forward layers evaluate): the
/// request page-rounded (16 KiB); from 1 MiB, rounded up to a step of an eighth of its power of two (at least a page),
/// at most 12.5 % more.
pub fn sizeClassBytes(bytes: u64) u64 {
    const page = @import("../../sdk/weights/streamed/phase_memory.zig").wire_page_bytes;
    const s_ = if (bytes > page) std.mem.alignForward(u64, bytes, page) else bytes;
    if (s_ < 1 << 20) return s_;
    const top: u64 = @as(u64, 1) << @intCast(std.math.log2_int(u64, s_));
    const step = @max(top / 8, page);
    return (std.math.divCeil(u64, s_, step) catch unreachable) * step;
}

/// An MLX buffer of `n` bytes as the Metal allocator takes it: past one page, rounded up to whole pages.
pub fn bufferBytes(n: u64, page: u64) u64 {
    return if (n > page) std.mem.alignForward(u64, n, page) else n;
}

/// The bytes the lanes hold at `cap` positions on every layer of `c` (`Cache.allocatedBytes` once every lane is
/// allocated), each buffer rounded up to `page`.
pub fn bytesAt(c: *const glm.Config, cap: u64, page: u64) u64 {
    const per_layer = bufferBytes(cap * 2 * c.kv_lora_rank, page) + bufferBytes(cap * 2 * c.qk_rope_head_dim, page);
    return c.n_layers * per_layer + c.nFull() * bufferBytes(cap * 2 * c.index_head_dim, page);
}

const testing = std.testing;

test "glm cache: the lanes' bytes at a cap are whole pages per buffer; at page multiples, the position's bytes times the cap" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    var c = try glm.Config.parse(testing.allocator, text, &glm.glm53, null);
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 95_232 * 16384), bytesAt(&c, 16384, 16384));
    // 17,412 positions: every lane's buffer rounds up to its next page.
    try testing.expectEqual(78 * (std.mem.alignForward(u64, 17412 * 1024, 16384) + std.mem.alignForward(u64, 17412 * 128, 16384)) + 21 * std.mem.alignForward(u64, 17412 * 256, 16384), bytesAt(&c, 17412, 16384));
    try testing.expect(bytesAt(&c, 17412, 16384) >= 95_232 * 17412);
    try testing.expectEqual(@as(u64, 100), bufferBytes(100, 16384));
}
