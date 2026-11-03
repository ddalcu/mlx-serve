//! The KV seam (R/mlx-stream/kv-seam-20261002.md): the lane storage a module-owned arch builds its decode state
//! from, generic over the arch's op backend `G` (`T`, `shapeOf`, `dtypeOf`, `keep`, `release`, `zeros`,
//! `hostArray`, `sliceUpdateDyn`, `slice`, `concat`). The lanes hold rows; they never compute attention. The route
//! is picked once at state construction and never changes during a request; a bounded lane is allocated once at
//! its cap and an append past it is refused by name before anything is written. No per-forward mark: `trim` is the
//! only in-request rewind.

const std = @import("std");

/// The widest array rank a lane slices (the backends' shape arrays hold at most this many dims).
pub const max_dims = 8;

/// Which backing the lanes use (Python precedence KV_BOUNDED > WINDOW_RING >
/// KV_CHUNK_GROW > plain), picked once when the sequence state is built.
pub const Route = enum {
    /// `_grow` concatenate stores: the stock path, every lever off.
    full_history,
    /// W73 `_GrowBuffer` lanes (geometric capacity from 256).
    chunk_grow,
    /// W80 `_WindowRing` window; compress / index `_GrowBuffer`s sized to
    /// `max_kv` when set, else geometric; the frontier a plain store.
    window_ring,
    /// W107: the ring plus compress / index / frontier preallocated to
    /// `max_kv` (geometric when unset), appends past the cap refused.
    bounded,
};

pub const Geometry = struct {
    route: Route = .full_history,
    /// The widest verify block one forward appends (DSpark depth + 1 fits).
    max_verify: u32 = 8,
    slack: u32 = 8,
    /// In-place appends between two ring compactions.
    headroom: u32 = 64,
    /// Preallocation capacity (`--max-kv`); null keeps geometric growth.
    max_kv: ?u32 = null,
};

/// `_BOUNDED_COMP_SLACK`, `_BOUNDED_LATENT_SLACK`.
const bounded_comp_slack = 8;
const bounded_latent_slack = 8;

/// `_bounded_comp_cap`: one row per completed group plus a verify margin.
pub fn boundedCompCap(max_kv: ?u32, ratio: u32) ?u32 {
    const m = max_kv orelse return null;
    const r = if (ratio > 0) ratio else 1;
    return (m + r - 1) / r + bounded_comp_slack;
}

/// `_bounded_latent_cap`: one fed row per token plus a verify margin.
pub fn boundedLatentCap(max_kv: ?u32) ?u32 {
    return (max_kv orelse return null) + bounded_latent_slack;
}

/// A request's lane bound (the arch's constants): the positions a request that declared no budget may still
/// generate, and the rows one forward may append past the last committed position (a verify block).
pub const Bound = struct { headroom: u64, scratch: u32 };

/// The positions a request's lanes are allocated at, once, at its prompt pass: its reservation (prompt plus
/// max_tokens when the shell declares one), else the prompt plus the headroom; plus the scratch rows.
pub fn capacity(prompt_tokens: u64, reserved_tokens: u64, b: Bound) u32 {
    const budget: u64 = if (reserved_tokens > prompt_tokens) reserved_tokens else prompt_tokens + b.headroom;
    return @intCast(budget + b.scratch);
}

/// The positions the bill charges for a request of `prompt_tokens` + `max_tokens`: the larger of the undeclared
/// (headroom) and the declared (reservation) allocation, so the bill bounds either form of the request at that
/// prompt. Kept apart from `capacity` by name: capacity is what one request allocates, billed capacity the bound.
pub fn billedCapacity(prompt_tokens: u64, max_tokens: u64, b: Bound) u32 {
    return @max(capacity(prompt_tokens, 0, b), capacity(prompt_tokens, prompt_tokens + max_tokens, b));
}

/// A `Ring` of `window` rows at `geo`: its base, the window plus a verify block, its slack and the headroom.
pub fn ringBase(window: u64, geo: Geometry) u64 {
    return window + geo.max_verify + geo.slack + geo.headroom;
}

/// A ring's rows through a prompt of `seq` positions fed in chunks of `chunk_rows` (the arch's chunk rule): both of its
/// slots at the compaction size (a chunk plus the window less one, at least the base), at every chunk count. That bounds
/// the ring at every instant: its first write holds its slot twice (the zeros the write consumes, and the write), a
/// compaction holds its source beside its destination, and from the third chunk both slots stay live. The bill adds
/// every layer's ring at the same instant, so only a per-ring bound at every instant is a global one.
pub fn ringPromptRows(window: u64, chunk_rows: u64, seq: u64, geo: Geometry) u64 {
    const chunk = @min(chunk_rows, @max(seq, 1));
    return 2 * @max(ringBase(window, geo), chunk + window -| 1);
}

/// A ring's rows in decode at its widest: the first step compacts the prompt's last chunk's ring (its rows plus the
/// window less one, at least the base) beside a new base; steady decode holds two bases.
pub fn ringDecodeRows(window: u64, chunk_rows: u64, seq: u64, geo: Geometry) u64 {
    const chunk = @min(chunk_rows, @max(seq, 1));
    const n_last = ((@max(seq, 1) - 1) % chunk) + 1;
    return @max(ringBase(window, geo), n_last + window -| 1) + ringBase(window, geo);
}

/// One bounded lane of a request's plan: the rows of its cap and the bytes of one row.
pub const LanePlan = struct { rows: u64, row_bytes: u64 };
/// One ring of a request's plan: its window and the bytes of one ring row (over every array that row spans).
pub const RingPlan = struct { window: u64, row_bytes: u64 };
/// A request's KV as its arch describes it: the bounded lanes, each allocated once at its cap, and the rings.
pub const Plan = struct { lanes: []const LanePlan, rings: []const RingPlan };
pub const Phase = enum { prompt, decode };

/// The plan's bounded lanes at their caps.
pub fn lanesBytes(p: Plan) u64 {
    var n: u64 = 0;
    for (p.lanes) |l| n += l.rows * l.row_bytes;
    return n;
}

/// The plan's KV at its widest in `phase` for a prompt of `seq` fed in chunks of `chunk_rows` (the bill's KV term):
/// the lanes at their caps and every ring at its phase's widest rows.
pub fn planBytes(p: Plan, phase: Phase, chunk_rows: u64, seq: u64, geo: Geometry) u64 {
    var n = lanesBytes(p);
    for (p.rings) |r| n += r.row_bytes * switch (phase) {
        .prompt => ringPromptRows(r.window, chunk_rows, seq, geo),
        .decode => ringDecodeRows(r.window, chunk_rows, seq, geo),
    };
    return n;
}

pub const Error = error{ BoundedLaneFull, RingRollbackTooDeep, TrimPastStart };

pub fn Lanes(comptime G: type) type {
    return struct {
        pub const T = G.T;

        /// `x[:, lo:hi]` along the sequence axis.
        pub fn rowsSlice(g: *G, x: T, lo: u32, hi: u32) !T {
            const s = g.shapeOf(x);
            var start: [max_dims]c_int = @splat(0);
            var stop: [max_dims]c_int = undefined;
            const strides: [max_dims]c_int = @splat(1);
            @memcpy(stop[0..s.n], s.slice());
            start[1] = @intCast(lo);
            stop[1] = @intCast(hi);
            return g.slice(x, start[0..s.n], stop[0..s.n], strides[0..s.n]);
        }

        /// `mx.slice_update(buf, new, mx.array([0, row, 0..], int32), axes=all)`.
        fn write(g: *G, buf: T, new: T, row: u32) !T {
            const n = g.shapeOf(new).n;
            var starts: [max_dims]i32 = @splat(0);
            starts[1] = @intCast(row);
            const sa = try g.hostArray(std.mem.sliceAsBytes(starts[0..n]), &.{@intCast(n)}, .int32);
            return g.sliceUpdateDyn(buf, new, sa);
        }

        /// `mx.zeros((b, cap) + tail, new.dtype)`.
        fn alloc(g: *G, like: T, cap: u32) !T {
            var s = g.shapeOf(like);
            s.d[1] = @intCast(cap);
            return g.zeros(s.slice(), g.dtypeOf(like));
        }

        fn replace(g: *G, slot: *?T, next: T) void {
            const kept = g.keep(next);
            if (slot.*) |old| g.release(old);
            slot.* = kept;
        }

        /// `_grow` / `_truncate`: a plain append-only store.
        pub const Concat = struct {
            rows: ?T = null,
            len: u32 = 0,

            pub fn append(self: *Concat, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.rows) |old| {
                    replace(g, &self.rows, try g.concat(&.{ old, new }, 1));
                } else replace(g, &self.rows, new);
                self.len += n;
            }

            pub fn view(self: *const Concat) ?T {
                return self.rows;
            }

            pub fn truncate(self: *Concat, g: *G, n: u32) !void {
                const rows = self.rows orelse return;
                if (n == 0) {
                    g.release(rows);
                    self.rows = null;
                    self.len = 0;
                    return;
                }
                if (n >= self.len) return;
                replace(g, &self.rows, try rowsSlice(g, rows, 0, n));
                self.len = n;
            }

            pub fn deinit(self: *Concat, g: *G) void {
                if (self.rows) |r| g.release(r);
                self.* = .{};
            }
        };

        /// `_GrowBuffer`: a capacity buffer with a logical length, written in
        /// place; `view` is byte-identical to the concatenated store.
        pub const Grow = struct {
            buf: ?T = null,
            len: u32 = 0,
            cap: u32 = 0,
            init_cap: u32 = 256,
            bounded_cap: ?u32 = null,

            pub fn init(init_cap: u32, bounded_cap: ?u32) Grow {
                return .{ .init_cap = if (bounded_cap) |b| b else init_cap, .bounded_cap = bounded_cap };
            }

            pub fn append(self: *Grow, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.bounded_cap) |b| if (self.len + n > b) return error.BoundedLaneFull;
                if (self.buf == null) {
                    const cap = @max(self.init_cap, n);
                    const buf = try alloc(g, new, cap);
                    replace(g, &self.buf, try write(g, buf, new, 0));
                    self.cap = cap;
                    self.len = n;
                    return;
                }
                if (self.len + n <= self.cap) {
                    replace(g, &self.buf, try write(g, self.buf.?, new, self.len));
                    self.len += n;
                    return;
                }
                const new_cap = @max(self.cap * 2, self.len + n);
                const head = try rowsSlice(g, self.buf.?, 0, self.len);
                var buf = try alloc(g, new, new_cap);
                buf = try write(g, buf, head, 0);
                buf = try write(g, buf, new, self.len);
                replace(g, &self.buf, buf);
                self.cap = new_cap;
                self.len += n;
            }

            pub fn view(self: *const Grow, g: *G) !?T {
                const buf = self.buf orelse return null;
                if (self.len == 0) return null;
                if (self.len == self.cap) return buf;
                return try rowsSlice(g, buf, 0, self.len);
            }

            /// Length only: the capacity stays for the next append.
            pub fn truncateTo(self: *Grow, n: u32) void {
                self.len = @min(n, self.len);
            }

            /// The bound becomes exactly `cap` rows (at least the rows it holds): an empty lane allocates `cap` at its
            /// next append; a lane with rows gets a `cap`-row buffer holding them (the caller evaluates `buf`, after
            /// which the old buffer is freed). No-op at the same capacity.
            pub fn resize(self: *Grow, g: *G, cap: u32) !void {
                if (cap < self.len) return error.BoundedLaneFull;
                self.init_cap = cap;
                self.bounded_cap = cap;
                const old = self.buf orelse return;
                if (cap == self.cap) return;
                if (self.len == 0) {
                    g.release(old);
                    self.buf = null;
                    self.cap = 0;
                    return;
                }
                const head = try rowsSlice(g, old, 0, self.len);
                replace(g, &self.buf, try write(g, try alloc(g, old, cap), head, 0));
                self.cap = cap;
            }

            pub fn deinit(self: *Grow, g: *G) void {
                if (self.buf) |b| g.release(b);
                self.buf = null;
                self.len = 0;
                self.cap = 0;
            }
        };

        /// `_WindowRing`: a contiguous suffix of the window history in two
        /// ping-pong buffers; row j of `view` is absolute position `drop + j`.
        pub const Ring = struct {
            window: u32,
            cap_keep: u32,
            phys_cap: u32,
            base_phys_cap: u32,
            bufs: [2]?T = .{ null, null },
            caps: [2]u32 = .{ 0, 0 },
            cur: u1 = 0,
            len: u32 = 0,
            drop: u32 = 0,

            pub fn init(window: u32, geo: Geometry) Ring {
                const keep = window + geo.max_verify + geo.slack;
                return .{ .window = window, .cap_keep = keep, .phys_cap = keep + geo.headroom, .base_phys_cap = keep + geo.headroom };
            }

            pub fn logicalLen(self: *const Ring) u32 {
                return self.drop + self.len;
            }

            pub fn append(self: *Ring, g: *G, new: T) !void {
                const n: u32 = @intCast(g.shapeOf(new).dim(1));
                if (n == 0) return;
                if (self.bufs[self.cur] == null) {
                    const cap = @max(self.base_phys_cap, n);
                    self.phys_cap = cap;
                    replace(g, &self.bufs[0], try alloc(g, new, cap));
                    replace(g, &self.bufs[1], try alloc(g, new, cap));
                    self.caps = .{ cap, cap };
                    replace(g, &self.bufs[self.cur], try write(g, self.bufs[self.cur].?, new, 0));
                    self.len = n;
                    self.drop = 0;
                    return;
                }
                if (self.len + n <= self.phys_cap) {
                    replace(g, &self.bufs[self.cur], try write(g, self.bufs[self.cur].?, new, self.len));
                    self.len += n;
                    return;
                }
                // Compaction: keep the newest `keep` rows (the oldest new query's
                // whole causal window) in the OTHER buffer, drop the rest.
                const l = self.drop + self.len;
                const l_new = l + n;
                const keep = @min(l_new, @max(self.cap_keep, n + (self.window - 1)));
                const new_drop = l_new - keep;
                const retained = l - new_drop;
                const src = self.bufs[self.cur].?;
                const target = @max(self.base_phys_cap, keep);
                const dst_idx: u1 = 1 - self.cur;
                var dst: T = undefined;
                if (self.bufs[dst_idx] == null or self.caps[dst_idx] != target) {
                    dst = try alloc(g, new, target);
                } else dst = self.bufs[dst_idx].?;
                if (retained > 0) dst = try write(g, dst, try rowsSlice(g, src, self.len - retained, self.len), 0);
                dst = try write(g, dst, new, retained);
                if (target != self.phys_cap) {
                    self.phys_cap = target;
                    // The other slot re-allocates at the next compaction.
                    if (self.bufs[self.cur]) |b| g.release(b);
                    self.bufs[self.cur] = null;
                    self.caps[self.cur] = 0;
                }
                replace(g, &self.bufs[dst_idx], dst);
                self.caps[dst_idx] = target;
                self.cur = dst_idx;
                self.len = retained + n;
                self.drop = new_drop;
            }

            pub fn view(self: *const Ring, g: *G) !?T {
                const buf = self.bufs[self.cur] orelse return null;
                if (self.len == 0) return null;
                if (self.len == self.caps[self.cur]) return buf;
                return try rowsSlice(g, buf, 0, self.len);
            }

            /// A rollback whose next query still has its whole window resident.
            /// Nothing dropped yet means every row is (Python's predicate also
            /// refuses lengths below window - 1 there, which no row requires).
            pub fn canTruncateToLength(self: *const Ring, n: u32) bool {
                return n == 0 or self.drop == 0 or n + 1 >= self.window + self.drop;
            }

            pub fn truncateToLength(self: *Ring, n: u32) !void {
                if (!self.canTruncateToLength(n)) return error.RingRollbackTooDeep;
                if (n <= self.drop) {
                    self.len = 0;
                    self.drop = n;
                    return;
                }
                self.len = @min(self.len, n - self.drop);
            }

            pub fn deinit(self: *Ring, g: *G) void {
                for (&self.bufs) |*b| if (b.*) |x| {
                    g.release(x);
                    b.* = null;
                };
                self.len = 0;
            }

            /// The ring as it stands, its current slot COPIED (a fresh buffer: later writes, donated or not, never reach
            /// it) and the other slot left empty (a compaction allocates it): `restore` puts it back whole. The caller
            /// evaluates the copy (`snapArray`) before the ring is written again.
            pub fn snapshot(self: *const Ring, g: *G) !Ring {
                var s = self.*;
                s.bufs = .{ null, null };
                s.caps = .{ 0, 0 };
                if (self.bufs[self.cur]) |b| {
                    s.bufs[self.cur] = g.keep(try write(g, try alloc(g, b, self.caps[self.cur]), b, 0));
                    s.caps[self.cur] = self.caps[self.cur];
                }
                return s;
            }

            /// The snapshot's buffer (null: an empty ring).
            pub fn snapArray(self: *const Ring) ?T {
                return self.bufs[self.cur];
            }

            /// Back to `s` (a `snapshot` of this ring), taking its buffer: the snapshot is spent.
            pub fn restore(self: *Ring, g: *G, s: Ring) void {
                self.deinit(g);
                self.* = s;
            }
        };

        /// One store lane under its route.
        pub const Store = union(enum) {
            concat: Concat,
            grow: Grow,

            pub fn append(self: *Store, g: *G, new: T) !void {
                return switch (self.*) {
                    inline else => |*l| l.append(g, new),
                };
            }

            pub fn view(self: *const Store, g: *G) !?T {
                return switch (self.*) {
                    .concat => |*l| l.view(),
                    .grow => |*l| try l.view(g),
                };
            }

            pub fn rows(self: *const Store) u32 {
                return switch (self.*) {
                    inline else => |*l| l.len,
                };
            }

            pub fn truncate(self: *Store, g: *G, n: u32) !void {
                switch (self.*) {
                    .concat => |*l| try l.truncate(g, n),
                    .grow => |*l| l.truncateTo(n),
                }
            }

            pub fn deinit(self: *Store, g: *G) void {
                switch (self.*) {
                    inline else => |*l| l.deinit(g),
                }
            }
        };

        pub const Window = union(enum) {
            store: Store,
            ring: Ring,

            pub fn append(self: *Window, g: *G, new: T) !void {
                return switch (self.*) {
                    inline else => |*l| l.append(g, new),
                };
            }

            pub fn view(self: *const Window, g: *G) !?T {
                return switch (self.*) {
                    .store => |*l| l.view(g),
                    .ring => |*l| l.view(g),
                };
            }

            /// Absolute position of view row 0 (0 unless the ring dropped rows).
            pub fn dropOffset(self: *const Window) u32 {
                return switch (self.*) {
                    .store => 0,
                    .ring => |r| r.drop,
                };
            }

            /// Rows appended so far, dropped ones included (the logical length).
            pub fn rows(self: *const Window) u32 {
                return switch (self.*) {
                    .store => |*l| l.rows(),
                    .ring => |r| r.logicalLen(),
                };
            }

            /// Whether `truncateTo(n)` keeps what a later read needs (a ring's rule; a store keeps every row).
            pub fn canTruncateTo(self: *const Window, n: u32) bool {
                return switch (self.*) {
                    .store => true,
                    .ring => |r| r.canTruncateToLength(n),
                };
            }

            /// Back to `n` rows (logical).
            pub fn truncateTo(self: *Window, g: *G, n: u32) !void {
                switch (self.*) {
                    .store => |*l| try l.truncate(g, n),
                    .ring => |*r| try r.truncateToLength(n),
                }
            }

            pub fn deinit(self: *Window, g: *G) void {
                switch (self.*) {
                    inline else => |*l| l.deinit(g),
                }
            }

            /// A window's state to come back to: a store's rows (it only grows, so a truncate restores it), a ring's
            /// `Ring.snapshot`.
            pub const Snap = union(enum) {
                store: u32,
                ring: Ring,

                pub fn array(self: *const Snap) ?T {
                    return switch (self.*) {
                        .store => null,
                        .ring => |*r| r.snapArray(),
                    };
                }

                pub fn deinit(self: *Snap, g: *G) void {
                    switch (self.*) {
                        .store => {},
                        .ring => |*r| r.deinit(g),
                    }
                }
            };

            pub fn snapshot(self: *const Window, g: *G) !Snap {
                return switch (self.*) {
                    .store => |*l| .{ .store = l.rows() },
                    .ring => |*r| .{ .ring = try r.snapshot(g) },
                };
            }

            /// Back to `s` (spent): a store truncated to its rows, a ring restored whole.
            pub fn restore(self: *Window, g: *G, s: Snap) !void {
                switch (self.*) {
                    .store => |*l| try l.truncate(g, s.store),
                    .ring => |*r| r.restore(g, s.ring),
                }
            }
        };
    };
}

test "sdk kv: the bounded caps are one row per fed token or completed group, plus the verify margin; unset stays geometric" {
    try std.testing.expectEqual(@as(?u32, 17424), boundedLatentCap(17416));
    try std.testing.expectEqual(@as(?u32, 4354 + 8), boundedCompCap(17416, 4));
    try std.testing.expectEqual(@as(?u32, 17416 + 8), boundedCompCap(17416, 0));
    try std.testing.expectEqual(@as(?u32, null), boundedCompCap(null, 4));
    try std.testing.expectEqual(@as(?u32, null), boundedLatentCap(null));
}

test "sdk kv: capacity is the reservation or the prompt plus the headroom, plus the scratch rows; the bill takes the larger" {
    const b: Bound = .{ .headroom = 8192, .scratch = 8 };
    try std.testing.expectEqual(@as(u32, 16384 + 8192 + 8), capacity(16384, 0, b));
    try std.testing.expectEqual(@as(u32, 16384 + 1024 + 8), capacity(16384, 16384 + 1024, b));
    try std.testing.expectEqual(@as(u32, 40000 + 8), capacity(32768, 40000, b));
    try std.testing.expectEqual(@as(u32, 24584), billedCapacity(16384, 1024, b));
    try std.testing.expectEqual(@as(u32, 16384 + 10000 + 8), billedCapacity(16384, 10000, b));
}

test "sdk kv: a ring's rows per phase (both slots at every chunk count; decode compacts the last chunk beside a base) and a plan's bytes" {
    // a 16,384-token prompt in chunks of 953: the window ring (128) and a frontier ring (2) at the default geometry
    try std.testing.expectEqual(@as(u64, 208), ringBase(128, .{}));
    try std.testing.expectEqual(@as(u64, 2 * (953 + 127)), ringPromptRows(128, 953, 16384, .{}));
    try std.testing.expectEqual(@as(u64, (183 + 127) + 208), ringDecodeRows(128, 953, 16384, .{}));
    try std.testing.expectEqual(@as(u64, 2 * (953 + 1)), ringPromptRows(2, 953, 16384, .{}));
    try std.testing.expectEqual(@as(u64, (183 + 1) + 82), ringDecodeRows(2, 953, 16384, .{}));
    // two slots at every chunk count: two chunks, and a prompt shorter than a chunk (one chunk of its own length)
    try std.testing.expectEqual(@as(u64, 2 * (953 + 127)), ringPromptRows(128, 953, 1000, .{}));
    try std.testing.expectEqual(@as(u64, 2 * 208), ringPromptRows(128, 953, 50, .{}));
    const p: Plan = .{ .lanes = &.{ .{ .rows = 10, .row_bytes = 3 }, .{ .rows = 4, .row_bytes = 5 } }, .rings = &.{.{ .window = 128, .row_bytes = 7 }} };
    try std.testing.expectEqual(@as(u64, 50), lanesBytes(p));
    try std.testing.expectEqual(@as(u64, 50 + 7 * 2160), planBytes(p, .prompt, 953, 16384, .{}));
    try std.testing.expectEqual(@as(u64, 50 + 7 * 518), planBytes(p, .decode, 953, 16384, .{}));
}
