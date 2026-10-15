//! The DFlash2 drafter of a Core AI bundle (`<bundle>/drafter`, Forge's `dflash2_coreai_drafter.py`): one
//! ANE call per speculative cycle. `draft` writes the context rows committed since the last call
//! (target tap features -> per-layer K / V) and runs the 8-row query block [anchor, mask x 7]; the host
//! keeps the K / V ring (slot = position % W, a W-token sliding window) and walks the 7 draft rows'
//! top-16 candidates through the predecessor / successor codebooks. `ctx64` ingests 64 prompt rows.

const std = @import("std");
const ffi = @import("../coreai_ffi.zig");
const log = @import("../log.zig");
const coreai = @import("coreai.zig");

const NEG: f16 = -1e4;
pub const TOP_K = 16;
pub const RANK = 256;

pub const Error = error{ BadDrafter, LoadFailed, RunFailed, OutOfMemory };

/// The drafter's RoPE as Forge writes it: angles in f64, cos / sin rounded to f32, then stored as f16.
pub fn ropeRow(out_cos: []f16, out_sin: []f16, pos: u64, theta: f64) void {
    const half = out_cos.len / 2;
    for (0..out_cos.len) |c| {
        const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * (c % half))) / @as(f64, @floatFromInt(out_cos.len)));
        const ang = @as(f64, @floatFromInt(pos)) * inv;
        out_cos[c] = @floatCast(@as(f32, @floatCast(@cos(ang))));
        out_sin[c] = @floatCast(@as(f32, @floatCast(@sin(ang))));
    }
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The 7-step selector walk: from `anchor`, each row picks the candidate maximizing
/// unary + successor[c] . (predecessor[prev] * hp[row]).
pub fn walk(anchor: u32, cand: []const [TOP_K]u32, unary: []const [TOP_K]f32, hp: []const [RANK]f32, pred_cb: []const u16, succ_cb: []const u16, out: []u32) void {
    var prev = anchor;
    for (cand, unary, hp, out) |cs, us, h, *dst| {
        var v: [RANK]f32 = undefined;
        const p = pred_cb[@as(usize, prev) * RANK ..][0..RANK];
        for (&v, p, h) |*x, pc, hv| x.* = bf16ToF32(pc) * hv;
        var best: usize = 0;
        var best_s: f32 = -std.math.inf(f32);
        for (cs, us, 0..) |c, u, j| {
            const sc = succ_cb[@as(usize, c) * RANK ..][0..RANK];
            var s: f32 = u;
            for (sc, v) |x, y| s += bf16ToF32(x) * y;
            if (s > best_s) {
                best_s = s;
                best = j;
            }
        }
        prev = cs[best];
        dst.* = prev;
    }
}

/// The additive attention mask of a draft call: columns [ring (W) | new rows (R) | block (T)], rows
/// `pitch` apart. A ring slot or a new row is visible to query position q when it holds a position
/// within `window` of q.
pub fn fillMask(mask: []f16, pitch: usize, cols: usize, slot_pos: []const i64, rows: []const i64, q0: i64, T: usize, window: i64) void {
    const W = slot_pos.len;
    const R = cols - W - T;
    for (0..T) |i| {
        const q = q0 + @as(i64, @intCast(i));
        const m = mask[i * pitch ..][0..cols];
        for (slot_pos, m[0..W]) |p, *x| x.* = if (p >= 0 and @abs(q - p) < window) 0 else NEG;
        for (m[W .. W + R], 0..) |*x, j| x.* = if (j < rows.len and @abs(q - rows[j]) < window) 0 else NEG;
        @memset(m[W + R ..], 0);
    }
}

const Row = struct { feat: []f16, pos: i64 };

pub const Drafter = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    api: *const ffi.Api,
    /// The drafter whose loaded functions and selector this one borrows (`slot`); null owns them.
    base: ?*Drafter,
    model: *anyopaque,
    /// The drafter's `.aimodel` package.
    path: []const u8,
    W: usize,
    T: usize,
    R: usize,
    RP: usize,
    feat_scale: f32,
    theta: f64,
    window: i64,
    layers: usize,
    feat_dim: usize,
    draft: Fn,
    ctx64: Fn,
    /// Ring K / V per layer (KV head, W, head dim): inputs of `draft`, written by the host.
    rings: [][2]coreai.Tensor,
    slot_pos: []i64,
    pending: std.ArrayList(Row) = .empty,
    sel_map: []align(std.heap.page_size_min) const u8,
    pred_cb: []const u16,
    succ_cb: []const u16,
    scratch: []coreai.Cand,

    const Fn = struct {
        f: coreai.Function,
        binding: *anyopaque,
        ins: []coreai.Named,
        outs: []coreai.Named,

        fn t(self: Fn, name: []const u8, out: bool) coreai.Tensor {
            return coreai.findNamed(if (out) self.outs else self.ins, name).?.t;
        }
    };

    pub fn open(allocator: std.mem.Allocator, io: std.Io, api: *const ffi.Api, dir: []const u8, vocab: u32) !*Drafter {
        const self = try allocator.create(Drafter);
        errdefer allocator.destroy(self);
        self.* = undefined;
        self.allocator = allocator;
        self.api = api;
        self.base = null;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();

        const pkg = try findPackage(io, a, dir);
        const meta = try readJson(io, a, try std.fs.path.join(a, &.{ dir, try std.mem.concat(a, u8, &.{ pkg[0 .. pkg.len - ".aimodel".len], ".json" }) }));
        self.W = try field(meta, "W");
        self.T = try field(meta, "T");
        self.R = try field(meta, "R");
        self.RP = try field(meta, "RP");
        self.feat_scale = @floatCast(coreai.num(meta.object.get("feat_scale")) orelse return error.BadDrafter);
        const cfg = try readJson(io, a, try std.fs.path.join(a, &.{ dir, "config.json" }));
        self.layers = try field(cfg, "num_hidden_layers");
        self.window = @intCast(try field(cfg, "sliding_window"));
        const rp = cfg.object.get("rope_parameters") orelse return error.BadDrafter;
        self.theta = coreai.num(if (rp == .object) rp.object.get("rope_theta") else null) orelse return error.BadDrafter;

        const pz = try a.dupeSentinel(u8, try std.fs.path.join(a, &.{ dir, pkg }), 0);
        self.path = pz;
        var e: ffi.ErrBuf = .{};
        self.model = api.model_load(pz, @backingInt(ffi.Compute.ane), 1, e.ptr(), e.cap()) orelse {
            log.err("[coreai] drafter load: {s}\n", .{e.text()});
            return error.LoadFailed;
        };
        errdefer api.release(self.model);
        self.draft = try self.function("draft");
        errdefer self.releaseFn(self.draft);
        self.ctx64 = try self.function("ctx64");
        errdefer self.releaseFn(self.ctx64);
        self.feat_dim = @intCast(self.draft.t("feat", false).shape[1]);
        // `propose` and `fill` hold the queued rows' positions and a RoPE row on the stack.
        const rope = @max(self.draft.t("q_cos", false).shape[1], self.draft.t("ctx_cos", false).shape[1], self.ctx64.t("ctx_cos", false).shape[1]);
        if (self.R > 64 or rope > 256) return error.BadDrafter;
        try self.openSelector(dir, vocab);
        errdefer std.posix.munmap(self.sel_map);
        try self.initState();
        log.info("[coreai] drafter ready: {s}, {d} layers, window {d}\n", .{ pkg, self.layers, self.window });
        return self;
    }

    /// Another drafter on `base`'s loaded functions and selector, with its own buffers and context.
    pub fn slot(base: *Drafter) !*Drafter {
        const self = try base.allocator.create(Drafter);
        errdefer base.allocator.destroy(self);
        self.* = base.*;
        self.base = base;
        self.arena = std.heap.ArenaAllocator.init(base.allocator);
        errdefer self.arena.deinit();
        self.draft = try self.instance(base.draft.f);
        errdefer self.releaseFn(self.draft);
        self.ctx64 = try self.instance(base.ctx64.f);
        errdefer self.releaseFn(self.ctx64);
        try self.initState();
        return self;
    }

    /// The rings (inputs of this drafter's `draft` binding), their slot positions, an empty queue.
    fn initState(self: *Drafter) !void {
        const a = self.arena.allocator();
        self.rings = try a.alloc([2]coreai.Tensor, self.layers);
        var name: [16]u8 = undefined;
        for (self.rings, 0..) |*r, i| {
            r[0] = self.draft.t(std.fmt.bufPrint(&name, "kc{d}", .{i}) catch unreachable, false);
            r[1] = self.draft.t(std.fmt.bufPrint(&name, "vc{d}", .{i}) catch unreachable, false);
        }
        self.slot_pos = try a.alloc(i64, self.W);
        self.pending = .empty;
        self.scratch = try a.alloc(coreai.Cand, TOP_K);
        self.reset();
    }

    pub fn deinit(self: *Drafter) void {
        self.releaseFn(self.draft);
        self.releaseFn(self.ctx64);
        if (self.base == null) {
            self.api.release(self.model);
            std.posix.munmap(self.sel_map);
        }
        self.clearPending();
        self.pending.deinit(self.allocator);
        self.arena.deinit();
        self.allocator.destroy(self);
    }

    fn function(self: *Drafter, name: []const u8) !Fn {
        const f = try coreai.loadFunctionWith(self.api, self.arena.allocator(), self.model, name);
        errdefer self.api.release(f.handle);
        return self.instance(f);
    }

    /// Buffers for `f`'s inputs and outputs, bound to it.
    fn instance(self: *Drafter, f: coreai.Function) !Fn {
        const ins = try coreai.buffersForWith(self.api, self.arena.allocator(), f.inputs);
        errdefer coreai.releaseNamed(self.api, ins);
        const outs = try coreai.buffersForWith(self.api, self.arena.allocator(), f.outputs);
        errdefer coreai.releaseNamed(self.api, outs);
        return .{ .f = f, .binding = try coreai.bindNamed(self.api, f, ins, outs), .ins = ins, .outs = outs };
    }

    pub fn bufferBytes(self: *const Drafter) u64 {
        return coreai.namedBytes(self.draft.ins) + coreai.namedBytes(self.draft.outs) + coreai.namedBytes(self.ctx64.ins) + coreai.namedBytes(self.ctx64.outs);
    }

    /// `f`'s binding and buffers, and the loaded function itself when this drafter owns it.
    fn releaseFn(self: *Drafter, f: Fn) void {
        self.api.release(f.binding);
        coreai.releaseNamed(self.api, f.ins);
        coreai.releaseNamed(self.api, f.outs);
        if (self.base == null) self.api.release(f.f.handle);
    }

    fn openSelector(self: *Drafter, dir: []const u8, vocab: u32) !void {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&pbuf, "{s}/selector.safetensors", .{dir}, 0) catch return error.BadDrafter;
        self.sel_map = coreai.mapFile(p) orelse return error.BadDrafter;
        errdefer std.posix.munmap(self.sel_map);
        const size = self.sel_map.len;
        if (size < 8) return error.BadDrafter;
        const hlen: usize = @intCast(std.mem.readInt(u64, self.sel_map[0..8], .little));
        if (8 + hlen > size) return error.BadDrafter;
        const h = std.json.parseFromSliceLeaky(std.json.Value, self.arena.allocator(), self.sel_map[8 .. 8 + hlen], .{}) catch return error.BadDrafter;
        if (h != .object) return error.BadDrafter;
        self.pred_cb = try codebook(h, self.sel_map, 8 + hlen, "candidate_selector.predecessor_codebook", vocab);
        self.succ_cb = try codebook(h, self.sel_map, 8 + hlen, "candidate_selector.successor_codebook", vocab);
    }

    pub fn reset(self: *Drafter) void {
        for (self.rings) |r| for (r) |t| t.zero();
        @memset(self.slot_pos, -1);
        self.clearPending();
    }

    /// Whether the rings or the queue hold any rows (an empty drafter snapshots as a reset).
    pub fn hasContext(self: *const Drafter) bool {
        if (self.pending.items.len > 0) return true;
        for (self.slot_pos) |p| if (p >= 0) return true;
        return false;
    }

    /// Bytes `save` writes: the rings, their slot positions, then the queued rows.
    pub fn snapLen(self: *const Drafter) usize {
        var n: usize = self.slot_pos.len * @sizeOf(i64) + @sizeOf(u32);
        for (self.rings) |r| for (r) |t| {
            n += t.nbytes;
        };
        return n + self.pending.items.len * (@sizeOf(i64) + self.feat_dim * @sizeOf(f16));
    }

    pub fn save(self: *const Drafter, out: []u8) void {
        var o: usize = 0;
        for (self.rings) |r| for (r) |t| put(out, &o, t.base[0..t.nbytes]);
        put(out, &o, std.mem.sliceAsBytes(self.slot_pos));
        put(out, &o, std.mem.asBytes(&@as(u32, @intCast(self.pending.items.len))));
        for (self.pending.items) |r| {
            put(out, &o, std.mem.asBytes(&r.pos));
            put(out, &o, std.mem.sliceAsBytes(r.feat));
        }
    }

    pub fn load(self: *Drafter, in: []const u8) !void {
        self.clearPending();
        var o: usize = 0;
        for (self.rings) |r| for (r) |t| get(in, &o, t.base[0..t.nbytes]);
        get(in, &o, std.mem.sliceAsBytes(self.slot_pos));
        var n: u32 = undefined;
        get(in, &o, std.mem.asBytes(&n));
        for (0..n) |_| {
            var pos: i64 = undefined;
            get(in, &o, std.mem.asBytes(&pos));
            const feat = try self.allocator.alloc(f16, self.feat_dim);
            errdefer self.allocator.free(feat);
            get(in, &o, std.mem.sliceAsBytes(feat));
            try self.pending.append(self.allocator, .{ .feat = feat, .pos = pos });
        }
    }

    fn put(out: []u8, o: *usize, bytes: []const u8) void {
        @memcpy(out[o.*..][0..bytes.len], bytes);
        o.* += bytes.len;
    }

    fn get(in: []const u8, o: *usize, bytes: []u8) void {
        @memcpy(bytes, in[o.*..][0..bytes.len]);
        o.* += bytes.len;
    }

    fn clearPending(self: *Drafter) void {
        for (self.pending.items) |r| self.allocator.free(r.feat);
        self.pending.clearRetainingCapacity();
    }

    /// Queue committed target feature rows (n x feat_dim, contiguous); flushed in RP-row `ctx64`
    /// calls, the last <= R go with the next draft call.
    pub fn addContext(self: *Drafter, feats: []const f16, p0: i64, n: usize) !void {
        for (0..n) |j| try self.pending.append(self.allocator, .{
            .feat = try self.allocator.dupe(f16, feats[j * self.feat_dim ..][0..self.feat_dim]),
            .pos = p0 + @as(i64, @intCast(j)),
        });
        while (self.pending.items.len > self.R) {
            const n_rows = @min(self.RP, self.pending.items.len - self.R);
            const rows = self.pending.items[0..n_rows];
            self.fill(self.ctx64, rows, self.RP);
            try self.run(self.ctx64);
            self.commit(self.ctx64, rows);
            for (rows) |r| self.allocator.free(r.feat);
            self.pending.replaceRangeAssumeCapacity(0, n_rows, &.{});
        }
    }

    fn fill(self: *Drafter, f: Fn, rows: []const Row, n: usize) void {
        const feat = f.t("feat", false);
        feat.zero();
        const cos = f.t("ctx_cos", false);
        const sin = f.t("ctx_sin", false);
        const d = @as(usize, @intCast(cos.shape[1]));
        for (0..n) |j| {
            const pos: i64 = if (j < rows.len) rows[j].pos else 0;
            var cb: [256]f16 = undefined;
            var sb: [256]f16 = undefined;
            ropeRow(cb[0..d], sb[0..d], @intCast(pos), self.theta);
            for (0..d) |c| {
                cos.f16s()[j * cos.stride(0) + c * cos.stride(1)] = cb[c];
                sin.f16s()[j * sin.stride(0) + c * sin.stride(1)] = sb[c];
            }
            if (j >= rows.len) continue;
            for (rows[j].feat, 0..) |v, c| feat.f16s()[c * feat.stride(1) + j * feat.stride(3)] = @floatCast(@as(f32, v) * self.feat_scale);
        }
    }

    /// The new rows' K / V into their ring slots, after the call that computed them.
    fn commit(self: *Drafter, f: Fn, rows: []const Row) void {
        var name: [16]u8 = undefined;
        for (self.rings, 0..) |r, i| for (r, [_][]const u8{ "k_new", "v_new" }) |ring, kind| {
            const src = f.t(std.fmt.bufPrint(&name, "{s}{d}", .{ kind, i }) catch unreachable, true);
            const heads: usize = @intCast(src.shape[0]);
            const hd: usize = @intCast(src.shape[2]);
            for (rows, 0..) |row, j| {
                const s: usize = @intCast(@mod(row.pos, @as(i64, @intCast(self.W))));
                for (0..heads) |h| for (0..hd) |d| {
                    ring.f16s()[h * ring.stride(0) + s * ring.stride(1) + d * ring.stride(2)] = src.f16s()[h * src.stride(0) + j * src.stride(1) + d * src.stride(2)];
                };
            }
        };
        for (rows) |row| self.slot_pos[@intCast(@mod(row.pos, @as(i64, @intCast(self.W))))] = row.pos;
    }

    fn run(self: *Drafter, f: Fn) !void {
        var e: ffi.ErrBuf = .{};
        if (self.api.run(1, &.{f.binding}, null, e.ptr(), e.cap()) != 0) {
            log.err("[coreai] drafter run: {s}\n", .{e.text()});
            return error.RunFailed;
        }
    }

    /// Draft T - 1 tokens after `anchor` (at position p0, not yet in the target).
    pub fn propose(self: *Drafter, anchor: u32, p0: i64, emb: []const f16, out: []u32) !void {
        std.debug.assert(self.pending.items.len <= self.R and out.len == self.T - 1);
        const rows = self.pending.items;
        const f = self.draft;
        self.fill(f, rows, self.R);
        const qc = f.t("q_cos", false);
        const qs = f.t("q_sin", false);
        const d = @as(usize, @intCast(qc.shape[1]));
        for (0..self.T) |i| {
            var cb: [256]f16 = undefined;
            var sb: [256]f16 = undefined;
            ropeRow(cb[0..d], sb[0..d], @intCast(p0 + @as(i64, @intCast(i))), self.theta);
            for (0..d) |c| {
                qc.f16s()[i * qc.stride(0) + c * qc.stride(1)] = cb[c];
                qs.f16s()[i * qs.stride(0) + c * qs.stride(1)] = sb[c];
            }
        }
        const mask = f.t("mask", false);
        var rp: [64]i64 = undefined;
        for (rows, 0..) |r, j| rp[j] = r.pos;
        const cols: usize = @intCast(mask.shape[1]);
        fillMask(mask.f16s()[0 .. (self.T - 1) * mask.stride(0) + cols], mask.stride(0), cols, self.slot_pos, rp[0..rows.len], p0, self.T, self.window);
        const anc = f.t("anchor", false);
        for (emb, 0..) |v, h| anc.f16s()[h * anc.stride(1)] = v;
        try self.run(f);
        self.commit(f, rows);
        self.clearPending();

        const logits = f.t("logits", true);
        const hp_t = f.t("hp", true);
        var cand: [8][TOP_K]u32 = undefined;
        var unary: [8][TOP_K]f32 = undefined;
        var hp: [8][RANK]f32 = undefined;
        const steps = self.T - 1;
        for (0..steps) |i| {
            const row = logitsRow(logits, i);
            const top = coreai.topK(row, TOP_K, self.scratch);
            for (top, 0..) |c, j| {
                cand[i][j] = c.id;
                unary[i][j] = c.logit;
            }
            for (0..RANK) |c| hp[i][c] = hp_t.f16s()[c * hp_t.stride(1) + (i + 1) * hp_t.stride(3)];
        }
        walk(anchor, cand[0..steps], unary[0..steps], hp[0..steps], self.pred_cb, self.succ_cb, out);
    }
};

/// Row i of the draft logits, contiguous along the vocabulary: (rows, V).
fn logitsRow(t: coreai.Tensor, i: usize) []const f16 {
    std.debug.assert(t.shape.len == 2 and t.stride(1) == 1);
    return t.f16s()[i * t.stride(0) ..][0..@intCast(t.shape[1])];
}

fn codebook(h: std.json.Value, map: []const u8, data0: usize, name: []const u8, vocab: u32) ![]const u16 {
    const e = h.object.get(name) orelse return error.BadDrafter;
    if (e != .object) return error.BadDrafter;
    const dt = e.object.get("dtype") orelse return error.BadDrafter;
    if (dt != .string or !std.mem.eql(u8, dt.string, "BF16")) return error.BadDrafter;
    const off = e.object.get("data_offsets") orelse return error.BadDrafter;
    if (off != .array or off.array.items.len != 2) return error.BadDrafter;
    const lo: usize = @intCast(off.array.items[0].integer);
    const hi: usize = @intCast(off.array.items[1].integer);
    if (hi - lo != @as(usize, vocab) * RANK * 2 or data0 + hi > map.len or (data0 + lo) % 2 != 0) return error.BadDrafter;
    const p: [*]const u16 = @ptrCast(@alignCast(map.ptr + data0 + lo));
    return p[0 .. @as(usize, vocab) * RANK];
}

fn findPackage(io: std.Io, a: std.mem.Allocator, dir: []const u8) ![]const u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return error.BadDrafter;
    defer d.close(io);
    var it = d.iterate();
    while (it.next(io) catch null) |e| if (std.mem.endsWith(u8, e.name, ".aimodel")) return a.dupe(u8, e.name);
    return error.BadDrafter;
}

fn readJson(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.json.Value {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22)) catch return error.BadDrafter;
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return error.BadDrafter;
    return if (v == .object) v else error.BadDrafter;
}

fn field(v: std.json.Value, name: []const u8) !usize {
    const x = v.object.get(name) orelse return error.BadDrafter;
    return if (x == .integer and x.integer > 0) @intCast(x.integer) else error.BadDrafter;
}

// ── tests ──

const testing = std.testing;

test "draft mask: a ring slot or new row is visible only within the window; the block always is" {
    const slot_pos = [_]i64{ -1, 10, 3, 12 }; // W = 4
    const rows = [_]i64{13}; // R = 2, one real row
    const T = 2;
    const cols = 4 + 2 + T;
    var mask: [T * cols]f16 = undefined;
    fillMask(&mask, cols, cols, &slot_pos, &rows, 14, T, 5);
    // q = 14: slot 10 (|4| < 5) and 12 visible, 3 too far, the empty slot never; row 13 visible, the padding row not.
    try testing.expectEqualSlices(f16, &.{ NEG, 0, NEG, 0, 0, NEG, 0, 0 }, mask[0..cols]);
    // q = 15: slot 10 is now 5 away.
    try testing.expectEqualSlices(f16, &.{ NEG, NEG, NEG, 0, 0, NEG, 0, 0 }, mask[cols..]);
}

test "bf16 widens by its top 16 bits" {
    try testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    try testing.expectEqual(@as(f32, -2.0), bf16ToF32(0xC000));
}

test "selector walk: each row picks by unary + successor . (predecessor[prev] * hp), chaining prev" {
    // vocab 4, rank padded to RANK with zeros; only component 0 matters.
    var pred: [4 * RANK]u16 = @splat(0);
    var succ: [4 * RANK]u16 = @splat(0);
    pred[1 * RANK] = 0x3F80; // pred[1] = e0
    pred[2 * RANK] = 0xBF80; // pred[2] = -e0
    succ[2 * RANK] = 0x4000; // succ[2] = 2 e0
    succ[3 * RANK] = 0x3F80; // succ[3] = e0
    var hp: [2][RANK]f32 = .{ @splat(0), @splat(0) };
    hp[0][0] = 1;
    hp[1][0] = 1;
    const cand = [2][TOP_K]u32{ @splat(3), @splat(3) };
    var c0 = cand;
    c0[0][0] = 2; // row 0: 2 (unary 0 + 2) beats 3 (unary 0.5 + 1)
    c0[1][0] = 2; // row 1 from prev 2: 2 scores 0 - 2, 3 scores 0.5 - 1: 3 wins
    var unary: [2][TOP_K]f32 = .{ @splat(-100), @splat(-100) };
    unary[0][0] = 0;
    unary[0][1] = 0.5;
    unary[1][0] = 0;
    unary[1][1] = 0.5;
    var out: [2]u32 = undefined;
    walk(1, &c0, &unary, &hp, &pred, &succ, &out);
    try testing.expectEqualSlices(u32, &.{ 2, 3 }, &out);
}

test "drafter rope rounds through f32 like Forge" {
    var c: [4]f16 = undefined;
    var s: [4]f16 = undefined;
    ropeRow(&c, &s, 3, 1e7);
    try testing.expectEqual(@as(f16, @floatCast(@as(f32, @floatCast(@cos(@as(f64, 3)))))), c[0]);
    try testing.expectEqual(c[0], c[2]);
    try testing.expectEqual(@as(f16, @floatCast(@as(f32, @floatCast(@sin(3.0 / std.math.pow(f64, 1e7, 0.5)))))), s[1]);
}

test "drafter snapshot: rings, slot positions and queued rows come back as saved" {
    var k = [_]f16{ 1, 2, 3, 4 };
    var v = [_]f16{ 5, 6, 7, 8 };
    const shape = [_]i64{ 1, 2, 2 };
    const strides = [_]i64{ 4, 2, 1 };
    var rings = [_][2]coreai.Tensor{.{
        .{ .base = @ptrCast(&k), .dtype = .f16, .shape = &shape, .strides = &strides, .nbytes = 8 },
        .{ .base = @ptrCast(&v), .dtype = .f16, .shape = &shape, .strides = &strides, .nbytes = 8 },
    }};
    var slot_pos = [_]i64{ 4, -1 };
    var d: Drafter = undefined;
    d.allocator = testing.allocator;
    d.rings = &rings;
    d.slot_pos = &slot_pos;
    d.feat_dim = 3;
    d.R = 8; // nothing flushes to `ctx64`
    d.pending = .empty;
    defer {
        d.clearPending();
        d.pending.deinit(testing.allocator);
    }
    try d.addContext(&.{ 0.25, 0.5, 0.75 }, 5, 1);
    try testing.expect(d.hasContext());
    const snap = try testing.allocator.alloc(u8, d.snapLen());
    defer testing.allocator.free(snap);
    d.save(snap);
    d.reset();
    try testing.expect(!d.hasContext());
    try d.load(snap);
    try testing.expectEqualSlices(f16, &.{ 1, 2, 3, 4 }, &k);
    try testing.expectEqualSlices(f16, &.{ 5, 6, 7, 8 }, &v);
    try testing.expectEqualSlices(i64, &.{ 4, -1 }, &slot_pos);
    try testing.expectEqual(@as(usize, 1), d.pending.items.len);
    try testing.expectEqual(@as(i64, 5), d.pending.items[0].pos);
    try testing.expectEqualSlices(f16, &.{ 0.25, 0.5, 0.75 }, d.pending.items[0].feat);
}
