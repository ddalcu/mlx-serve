//! Multi-Mac inference over MLX's distributed group (JACCL = RDMA over
//! Thunderbolt 5, macOS 26.2+). Rank 0 is the server; every other rank is a
//! worker that executes what rank 0 tells it, in lockstep.
const std = @import("std");
const mlx = @import("mlx.zig");
const crt = @import("crt.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const status = @import("status.zig");
const transformer_mod = @import("transformer.zig");

pub const Mode = enum {
    /// Each rank owns a contiguous range of layers; the hidden stream hops at the seams.
    pipeline,
    /// Every rank runs every layer on its slice of each MoE expert; one all_sum per MoE layer.
    tensor,

    pub fn parse(s: []const u8) ?Mode {
        return std.meta.stringToEnum(Mode, s);
    }
};

/// How this process joins the group. Each rank names only its OWN RDMA device
/// (the `rdma_enX` whose port faces the peer); JACCL reads only its own row of
/// the device matrix but validates the whole square, so the other rows repeat it.
pub const Config = struct {
    rank: u32,
    size: u32 = 2,
    /// `ip:port` rank 0 listens on; workers connect out (no inbound firewall prompt there).
    coordinator: []const u8,
    device: []const u8,
};

/// Writes the JACCL device matrix for `cfg` into `out`.
pub fn deviceMatrixJson(cfg: Config, out: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(out);
    try w.writeByte('[');
    for (0..cfg.size) |r| {
        if (r != 0) try w.writeByte(',');
        try w.writeByte('[');
        for (0..cfg.size) |c| {
            if (c != 0) try w.writeByte(',');
            if (r == c) try w.writeAll("null") else try w.print("\"{s}\"", .{cfg.device});
        }
        try w.writeByte(']');
    }
    try w.writeByte(']');
    return w.buffered();
}

/// The trunk layers one pipeline rank holds: `[start, end)` of `total`. The rank
/// loads them renumbered from 0, so its config, caches and bills are those of a
/// short model; the hidden stream crosses ranks at `end`.
pub const LayerSpan = struct {
    start: u32,
    end: u32,
    total: u32,

    pub fn len(self: LayerSpan) u32 {
        return self.end - self.start;
    }

    /// Rank 0 embeds the tokens, projects the logits and runs the MTP head and vision
    /// tower; a later rank holds only its trunk layers.
    pub fn first(self: LayerSpan) bool {
        return self.start == 0;
    }

    /// The name tensor `key` loads under on this rank, or null when it belongs to
    /// another: `key` itself, or the renumbered trunk name written into `buf`.
    pub fn remap(self: LayerSpan, key: []const u8, buf: []u8) ?[]const u8 {
        if (headOnly(key)) return if (self.first()) key else null;
        const at = trunkLayerAt(key) orelse return key;
        const n = std.fmt.parseInt(u32, key[at.lo..at.hi], 10) catch return key;
        if (n < self.start or n >= self.end) return null;
        if (self.start == 0) return key;
        return std.fmt.bufPrint(buf, "{s}{d}{s}", .{ key[0..at.lo], n - self.start, key[at.hi..] }) catch null;
    }
};

/// Tensor-parallel rank: its slice of every MoE expert's intermediate width (gate/up
/// rows, down columns), the routed experts and the shared one alike.
pub const Shard = struct {
    rank: u32,
    size: u32,

    /// Which projection of an expert `key` names: gate/up split their rows, down its columns.
    pub fn expertAxis(key: []const u8) ?enum { rows, cols } {
        if (headOnly(key)) return null;
        if (std.mem.indexOf(u8, key, ".switch_mlp.") == null and std.mem.indexOf(u8, key, ".shared_expert.") == null) return null;
        const tail = for ([_][]const u8{ ".weight", ".scales", ".biases" }) |sfx| {
            if (std.mem.endsWith(u8, key, sfx)) break key[0 .. key.len - sfx.len];
        } else return null;
        if (std.mem.endsWith(u8, tail, ".gate_proj") or std.mem.endsWith(u8, tail, ".up_proj")) return .rows;
        if (std.mem.endsWith(u8, tail, ".down_proj")) return .cols;
        return null;
    }

    /// Copies this rank's part of axis `mid` of a row-major `[outer, mid, inner]` buffer.
    pub fn slice(self: Shard, out: []u8, buf: []const u8, outer: usize, mid: usize, inner: usize) void {
        const part = mid / self.size;
        for (0..outer) |o| {
            const src = (o * mid + self.rank * part) * inner;
            @memcpy(out[o * part * inner ..][0 .. part * inner], buf[src..][0 .. part * inner]);
        }
    }
};

/// Tensors only rank 0 reads: the MTP head and the vision/audio towers.
pub fn headOnly(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "mtp.") or std.mem.indexOf(u8, key, ".mtp.") != null or
        std.mem.indexOf(u8, key, "visual") != null or std.mem.indexOf(u8, key, "vision") != null or
        std.mem.indexOf(u8, key, "audio") != null;
}

/// The digits of the first `layers.<N>.` in `key`.
fn trunkLayerAt(key: []const u8) ?struct { lo: usize, hi: usize } {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, key, from, "layers.")) |i| {
        from = i + 1;
        if (i > 0 and key[i - 1] != '.') continue;
        const lo = i + "layers.".len;
        var hi = lo;
        while (hi < key.len and std.ascii.isDigit(key[hi])) hi += 1;
        if (hi > lo and hi < key.len and key[hi] == '.') return .{ .lo = lo, .hi = hi };
    }
    return null;
}

pub const Link = struct {
    group: mlx.mlx_distributed_group,
    rank: u32,
    size: u32,

    /// Joins the group. Blocks until every rank has connected to the coordinator.
    pub fn open(io: std.Io, cfg: Config) !Link {
        if (!mlx.mlx_distributed_is_available("jaccl")) {
            log.err("[dist] JACCL is unavailable: RDMA over Thunderbolt needs macOS 26.2+ and `rdma_ctl enable` (Recovery)\n", .{});
            return error.RdmaUnavailable;
        }
        var path_buf: [128]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/mlx-serve-jaccl-{d}.json", .{std.c.getpid()}, 0);
        var json_buf: [1024]u8 = undefined;
        const json = try deviceMatrixJson(cfg, &json_buf);
        {
            const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
            defer f.close(io);
            var wb: [256]u8 = undefined;
            var fw = f.writer(io, &wb);
            try fw.interface.writeAll(json);
            try fw.interface.flush();
        }
        var rank_buf: [16]u8 = undefined;
        var coord_buf: [128]u8 = undefined;
        crt.setEnv("MLX_RANK", try std.fmt.bufPrintSentinel(&rank_buf, "{d}", .{cfg.rank}, 0));
        crt.setEnv("MLX_JACCL_COORDINATOR", try std.fmt.bufPrintSentinel(&coord_buf, "{s}", .{cfg.coordinator}, 0));
        crt.setEnv("MLX_IBV_DEVICES", path);
        log.info("[dist] joining as rank {d}/{d} via {s}, coordinator {s}\n", .{ cfg.rank, cfg.size, cfg.device, cfg.coordinator });
        var g = mlx.mlx_distributed_group_new();
        mlx.check(mlx.mlx_distributed_init(&g, true, "jaccl")) catch {
            var eb: [512]u8 = undefined;
            log.err("[dist] JACCL init failed: {s}\n", .{mlx.takeError(&eb) orelse "unknown"});
            return error.RdmaInitFailed;
        };
        const link: Link = .{ .group = g, .rank = @intCast(mlx.mlx_distributed_group_rank(g)), .size = @intCast(mlx.mlx_distributed_group_size(g)) };
        if (link.rank != cfg.rank or link.size != cfg.size) return error.RdmaGroupMismatch;
        log.info("[dist] joined: rank {d} of {d}\n", .{ link.rank, link.size });
        return link;
    }

    pub fn deinit(self: *Link) void {
        _ = mlx.mlx_distributed_group_free(self.group);
    }

    // The collectives have CPU kernels only, so every op runs on this thread's CPU stream
    // (streams are thread-local); MLX fences the GPU work they read.

    /// Lazy send of `x` to `dst`; the returned array must be evaluated (or depended on).
    pub fn send(self: *const Link, x: mlx.mlx_array, dst: u32) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_send(&r, x, @intCast(dst), self.group, cs));
        return r;
    }

    pub fn recv(self: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_recv(&r, shape.ptr, shape.len, dtype, @intCast(src), self.group, cs));
        return r;
    }

    pub fn allSum(self: *const Link, x: mlx.mlx_array) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_all_sum(&r, x, self.group, cs));
        return r;
    }
};

// ── Wire protocol: fixed int32[8] control messages, then any payload tensor ──

const MAGIC: i32 = 0x4D4C5853; // "MLXS"
const VERSION: i32 = 1;
const Msg = [8]i32;
const OP_FORWARD: i32 = 1;
const OP_STOP: i32 = 2;
const OP_MOE: i32 = 3;

fn sendMsg(link: *const Link, m: Msg, dst: u32) !void {
    const shape = [_]c_int{8};
    const a = mlx.mlx_array_new_data(&m, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvMsg(link: *const Link, src: u32) !Msg {
    const r = try link.recv(&.{8}, .int32, src);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    const d = mlx.mlx_array_data_int32(r) orelse return error.DistBadMessage;
    var m: Msg = undefined;
    @memcpy(&m, d[0..8]);
    return m;
}

fn mib(b: u64) i32 {
    return @intCast(@min(b >> 20, std.math.maxInt(i32)));
}

/// What a worker reports on joining: its memory.
const Hello = struct {
    avail_mib: i32,
    total_mib: i32,

    fn here() Hello {
        return .{ .avail_mib = mib(status.getAvailableMemBytes()), .total_mib = mib(status.getTotalMemBytes()) };
    }
    fn msg(h: Hello) Msg {
        return .{ MAGIC, VERSION, h.avail_mib, h.total_mib, 0, 0, 0, 0 };
    }
    fn parse(m: Msg) !Hello {
        if (m[0] != MAGIC or m[1] != VERSION) return error.DistProtocolMismatch;
        return .{ .avail_mib = m[2], .total_mib = m[3] };
    }
};

/// What rank 0 asks a worker to load: the model by folder name, checked against
/// the worker's own copy by geometry, and its part of it.
const Plan = struct {
    mode: Mode,
    span: LayerSpan,
    hidden: i32,
    experts: i32,

    fn msg(p: Plan, name_len: usize) Msg {
        return .{ MAGIC, @intFromEnum(p.mode), @intCast(p.span.start), @intCast(p.span.end), @intCast(p.span.total), @intCast(name_len), p.hidden, p.experts };
    }
    fn parse(m: Msg) !Plan {
        if (m[0] != MAGIC) return error.DistProtocolMismatch;
        return .{ .mode = @enumFromInt(m[1]), .span = .{ .start = @intCast(m[2]), .end = @intCast(m[3]), .total = @intCast(m[4]) }, .hidden = m[6], .experts = m[7] };
    }
    fn matches(p: Plan, cfg: *const model_mod.ModelConfig) bool {
        return cfg.num_hidden_layers == p.span.total and cfg.hidden_size == p.hidden and cfg.num_experts == p.experts;
    }
};

/// Worker failure codes in its READY message (0 = ready).
const NOT_FOUND: i32 = 2;
const MISMATCH: i32 = 3;
const LOAD_FAILED: i32 = 1;

/// The seam that splits `layers` by free memory, on a full-attention group boundary.
pub fn chooseSplit(layers: u32, interval: u32, avail0: u64, avail1: u64) u32 {
    const g = @max(interval, 1);
    const share = @as(f64, @floatFromInt(avail0)) / @as(f64, @floatFromInt(@max(avail0 + avail1, 1)));
    const raw: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(layers)) * share / @as(f64, @floatFromInt(g))));
    return std.math.clamp(raw, 1, layers / g - 1) * g;
}

/// Rank 0's handle on the worker: the group, the split, and how far the worker's caches have advanced.
pub const Peer = struct {
    link: Link,
    mode: Mode,
    /// The model the group serves, by folder name (`--dist-model`, else `--model`).
    model: []const u8,
    split: ?u32,
    worker: Hello,
    /// Set by the first load of `model` (`planFor`), reused by every reload.
    span: LayerSpan = .{ .start = 0, .end = 0, .total = 0 },
    shard: ?Shard = null,
    planned: bool = false,
    ready: bool = false,
    /// Exchanges with the worker since the group formed (`/props` cluster).
    exchanges: u64 = 0,
    /// Tokens the worker's caches hold. Every forward must continue them or start over at 0.
    peer_pos: usize = 0,

    /// Hands the stream to the worker and returns its output stream. Lazy: the sends are
    /// queued (async) ahead of the receive so the comm stream runs them in that order.
    pub fn hop(self: *Peer, h: mlx.mlx_array, offset: usize) !mlx.mlx_array {
        if (!self.ready) try self.awaitReady();
        const sh = mlx.getShape(h);
        log.debug("[dist] hop at {d}: {d}x{d} rows (worker holds {d})\n", .{ offset, sh[0], sh[1], self.peer_pos });
        if (offset != 0 and offset != self.peer_pos) {
            log.err("[dist] forward at {d} but the worker holds {d} tokens: its caches cannot follow this reuse\n", .{ offset, self.peer_pos });
            return error.DistPeerDiverged;
        }
        try self.sendOrdered(.{ OP_FORWARD, sh[0], sh[1], @intCast(offset & 0x7fffffff), @intCast(offset >> 31), sh[2], @intFromEnum(mlx.mlx_array_dtype(h)), 0 }, &.{h});
        self.peer_pos = offset + @as(usize, @intCast(sh[1]));
        self.exchanges += 1;
        return queuedRecv(&self.link, sh, mlx.mlx_array_dtype(h), 1);
    }

    /// Tensor mode: one MoE layer's input and router logits to the worker.
    pub fn sendMoe(self: *Peer, x: mlx.mlx_array, logits: mlx.mlx_array, layer: u32) !void {
        if (!self.ready) try self.awaitReady();
        const xs = mlx.getShape(x);
        const ls = mlx.getShape(logits);
        self.exchanges += 1;
        try self.sendOrdered(.{ OP_MOE, xs[0], xs[1], xs[2], ls[ls.len - 1], @intFromEnum(mlx.mlx_array_dtype(x)), @intCast(layer), @intFromEnum(mlx.mlx_array_dtype(logits)) }, &.{ x, logits });
    }

    /// The worker's partial for the layer input `x`: its shape and dtype.
    pub fn recvLike(self: *Peer, x: mlx.mlx_array) !mlx.mlx_array {
        return queuedRecv(&self.link, mlx.getShape(x), mlx.mlx_array_dtype(x), 1);
    }

    /// Header, then each payload, chained so the comm stream runs them in order, and
    /// queued now (async) so a receive built after them cannot overtake them.
    fn sendOrdered(self: *Peer, hdr_host: Msg, payload: []const mlx.mlx_array) !void {
        const shape = [_]c_int{8};
        const hdr = mlx.mlx_array_new_data(&hdr_host, &shape, 1, .int32);
        defer _ = mlx.mlx_array_free(hdr);
        var last = try self.link.send(hdr, 1);
        defer _ = mlx.mlx_array_free(last);
        for (payload) |p| {
            const ordered = try dependsOn(p, last);
            defer _ = mlx.mlx_array_free(ordered);
            const sent = try self.link.send(ordered, 1);
            _ = mlx.mlx_array_free(last);
            last = sent;
        }
        const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{last}, 1);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }

    fn awaitReady(self: *Peer) !void {
        const m = try recvMsg(&self.link, 1);
        if (m[0] != MAGIC or m[1] != 0) {
            switch (m[1]) {
                NOT_FOUND => log.err("[dist] the worker has no model named {s} in its model folders\n", .{self.model}),
                MISMATCH => log.err("[dist] the worker's copy of {s} has a different geometry\n", .{self.model}),
                else => log.err("[dist] the worker failed to load its part of {s}\n", .{self.model}),
            }
            return error.DistWorkerLoadFailed;
        }
        if (self.mode == .pipeline)
            log.info("[dist] worker ready: layers {d}..{d} resident, {d} MiB\n", .{ self.span.end, self.span.total, m[2] })
        else
            log.info("[dist] worker ready: its half of every expert resident, {d} MiB\n", .{m[2]});
        self.ready = true;
    }
};

/// A receive completed on the host before anything reads it. A lazy one lets a GPU command
/// buffer wait on it, and MLX signals a buffer's events only when the whole buffer ends: the
/// buffer that computes the payload can then also hold the wait for the reply to it, which
/// deadlocks until Metal's watchdog kills it. Done here, the receive also pairs with the
/// peer's send even when nobody reads it (a warmup that drops its output).
fn queuedRecv(link: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
    const r = try link.recv(shape, dtype, src);
    errdefer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    return r;
}

/// `x` that evaluates only after `dep`.
fn dependsOn(x: mlx.mlx_array, dep: mlx.mlx_array) !mlx.mlx_array {
    const ins = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(ins);
    const deps = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dep}, 1);
    defer _ = mlx.mlx_vector_array_free(deps);
    var out = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_depends(&out, ins, deps));
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&r, out, 0));
    return r;
}

var peer_slot: ?Peer = null;

/// The group this server leads (`--dist`): set at startup, while a thread waits
/// for the worker to join; the handshake runs when the group's model loads.
const Group = struct { mode: Mode, model: []const u8, split: ?u32 };
var group: ?Group = null;
var joined = std.atomic.Value(bool).init(false);
var joined_link: Link = undefined;

/// The `/props` "cluster" object (with its leading comma), empty when this server is
/// not rank 0 of a group: what the app draws the split from.
pub fn propsJson(allocator: std.mem.Allocator) ![]u8 {
    if (peer()) |p| return clusterJson(allocator, p);
    const g = group orelse return allocator.dupe(u8, "");
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":\"{s}\",\"joined\":{},\"planned\":false,\"worker_ready\":false}}", .{
        @tagName(g.mode), std.fs.path.basename(g.model), joined.load(.acquire),
    });
}

fn clusterJson(allocator: std.mem.Allocator, p: *const Peer) ![]u8 {
    const sp = p.span;
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":\"{s}\",\"joined\":true,\"planned\":{},\"worker_ready\":{},\"layers\":{d},\"here\":[{d},{d}],\"worker\":[{d},{d}],\"expert_shards\":{d},\"worker_free_mib\":{d},\"worker_total_mib\":{d},\"exchanges\":{d}}}", .{
        @tagName(p.mode), std.fs.path.basename(p.model), p.planned, p.ready, sp.total, sp.start, sp.end,
        if (p.mode == .pipeline) sp.end else 0, sp.total, if (p.shard) |sh| sh.size else 0,
        p.worker.avail_mib, p.worker.total_mib, p.exchanges,
    });
}

/// Rank 0 at shutdown: releases the worker. A peer that vanishes mid-protocol crashes the
/// other side inside Apple's RDMA library, so the group always ends with this message.
pub fn shutdown() void {
    const p = peer() orelse return;
    if (!p.ready) return;
    sendMsg(&p.link, .{ OP_STOP, 0, 0, 0, 0, 0, 0, 0 }, 1) catch return;
    log.info("[dist] worker released\n", .{});
}

/// Rank 0's peer, once `connect` has run.
pub fn peer() ?*Peer {
    return if (peer_slot) |*p| p else null;
}

/// What this rank holds of `model_dir`'s model when it is the group's model: on the first
/// load the split is chosen and the worker told to load its part; reloads reuse it.
pub const Part = union(enum) { span: LayerSpan, shard: Shard };

pub fn planFor(model_dir: []const u8, config: *const model_mod.ModelConfig) !?Part {
    const g = group orelse return null;
    if (!std.mem.eql(u8, std.fs.path.basename(g.model), std.fs.path.basename(model_dir))) return null;
    if (peer_slot == null) {
        if (!joined.load(.acquire)) {
            log.err("[dist] {s} is served with a worker: start `mlx-serve worker` on the other Mac first\n", .{std.fs.path.basename(g.model)});
            return error.DistWorkerNotJoined;
        }
        const worker = try Hello.parse(try recvMsg(&joined_link, 1));
        log.info("[dist] worker: {d} MiB free of {d}\n", .{ worker.avail_mib, worker.total_mib });
        peer_slot = .{ .link = joined_link, .mode = g.mode, .model = g.model, .split = g.split, .worker = worker };
    }
    const p = &peer_slot.?;
    if (!p.planned) try plan(p, config);
    return if (p.shard) |sh| .{ .shard = sh } else .{ .span = p.span };
}

fn plan(self: *Peer, config: *const model_mod.ModelConfig) !void {
    if (!config.isQwen4()) {
        log.err("[dist] {s}: this architecture ({s}) has no multi-Mac path yet\n", .{ self.model, config.model_type });
        return error.DistArchUnsupported;
    }
    const n = config.num_hidden_layers;
    var pl: Plan = .{ .mode = self.mode, .span = .{ .start = 0, .end = n, .total = n }, .hidden = @intCast(config.hidden_size), .experts = @intCast(config.num_experts) };
    switch (self.mode) {
        .tensor => {
            if (config.num_experts == 0) return error.DistTensorNeedsMoe;
            self.shard = .{ .rank = 0, .size = 2 };
            log.info("[dist] tensor: each expert of {s} split across this Mac and the worker\n", .{std.fs.path.basename(self.model)});
        },
        .pipeline => {
            const mine: u64 = status.getAvailableMemBytes() >> 20;
            const k = self.split orelse chooseSplit(n, config.full_attention_interval, mine, @intCast(self.worker.avail_mib));
            if (k == 0 or k >= n or (config.full_attention_interval != 0 and k % config.full_attention_interval != 0)) return error.DistSplitOffInterval;
            self.span = .{ .start = 0, .end = k, .total = n };
            pl.span = .{ .start = k, .end = n, .total = n };
            log.info("[dist] pipeline: layers 0..{d} of {s} here ({d} MiB free), {d}..{d} on the worker ({d} MiB free of {d})\n", .{ k, std.fs.path.basename(self.model), mine, k, n, self.worker.avail_mib, self.worker.total_mib });
        },
    }
    const name = std.fs.path.basename(self.model);
    try sendMsg(&self.link, pl.msg(name.len), 1);
    try sendBytes(&self.link, name, 1);
    self.planned = true;
}

fn sendBytes(link: *const Link, bytes: []const u8, dst: u32) !void {
    const a = mlx.mlx_array_new_data(bytes.ptr, &[_]c_int{@intCast(bytes.len)}, 1, .uint8);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvBytes(link: *const Link, out: []u8, src: u32) !void {
    const r = try link.recv(&[_]c_int{@intCast(out.len)}, .uint8, src);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    @memcpy(out, (mlx.mlx_array_data_uint8(r) orelse return error.DistBadMessage)[0..out.len]);
}

/// Rank 0 at startup: waits for the worker on a thread of its own, so the server
/// serves meanwhile. Only the RDMA link opens there; every message runs on the
/// inference thread, when `model` loads (`planFor`).
pub fn connect(io: std.Io, allocator: std.mem.Allocator, cfg: Config, mode: Mode, model: []const u8, split: ?u32) !void {
    group = .{ .mode = mode, .model = try allocator.dupe(u8, model), .split = split };
    const t = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, waitForWorker, .{ io, cfg });
    t.detach();
}

fn waitForWorker(io: std.Io, cfg: Config) void {
    joined_link = Link.open(io, cfg) catch |err| {
        log.err("[dist] the cluster link did not open: {s}\n", .{@errorName(err)});
        return;
    };
    joined.store(true, .release);
    log.info("[dist] worker joined; it loads its part with {s}\n", .{std.fs.path.basename(group.?.model)});
}

/// The folder named `name` in `roots` (`<root>/<name>` or `<root>/<org>/<name>`).
fn findModel(io: std.Io, allocator: std.mem.Allocator, roots: []const []const u8, name: []const u8) !?[]u8 {
    for (roots) |root| {
        const direct = try std.fs.path.join(allocator, &.{ root, name, "config.json" });
        defer allocator.free(direct);
        if (std.Io.Dir.cwd().statFile(io, direct, .{})) |_| return try std.fs.path.join(allocator, &.{ root, name }) else |_| {}
        var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |e| {
            if (e.kind != .directory) continue;
            const p = try std.fs.path.join(allocator, &.{ root, e.name, name, "config.json" });
            defer allocator.free(p);
            if (std.Io.Dir.cwd().statFile(io, p, .{})) |_| return try std.fs.path.join(allocator, &.{ root, e.name, name }) else |_| {}
        }
    }
    return null;
}

/// `mlx-serve worker --coordinator IP:PORT --device rdma_enX [--model-dir DIR]...`: joins
/// rank 0's group, loads the part of the model rank 0 names (found by folder name in its own
/// model folders, `~/.mlx-serve/models` by default), then runs forwards until rank 0 stops it.
pub fn cmdWorker(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !void {
    var roots: std.ArrayList([]const u8) = .empty;
    defer roots.deinit(allocator);
    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(allocator);
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model-dir") and i + 1 < args.len) {
            try roots.append(allocator, args[i + 1]);
            i += 1;
        } else try rest.append(allocator, args[i]);
    }
    var home_root: ?[]u8 = null;
    defer if (home_root) |h| allocator.free(h);
    if (roots.items.len == 0) {
        home_root = try @import("cli.zig").modelsRootPath(allocator, std.mem.span(std.c.getenv("HOME") orelse "/tmp"));
        try roots.append(allocator, home_root.?);
    }
    try rest.appendSlice(allocator, &.{ "--rank", "1" });
    const cfg = try parseArgs(rest.items);

    // The host listens only once its server is up: keep knocking until it is.
    var link = while (true) break Link.open(io, cfg) catch |err| switch (err) {
        error.RdmaInitFailed => {
            log.info("[dist] worker: the host at {s} is not listening yet; retrying\n", .{cfg.coordinator});
            std.Io.sleep(io, .fromSeconds(3), .real) catch {};
            continue;
        },
        else => return err,
    };
    defer link.deinit();
    try sendMsg(&link, Hello.here().msg(), 0);
    log.info("[dist] worker: joined, waiting for the host to load a model\n", .{});
    const pmsg = try recvMsg(&link, 0);
    const pl = try Plan.parse(pmsg);
    var name_buf: [256]u8 = undefined;
    const name_len: usize = @intCast(pmsg[5]);
    if (name_len == 0 or name_len > name_buf.len) return error.DistBadMessage;
    try recvBytes(&link, name_buf[0..name_len], 0);
    const name = name_buf[0..name_len];
    if (pl.mode == .pipeline)
        log.info("[dist] worker: {s} layers {d}..{d} of {d} (pipeline)\n", .{ name, pl.span.start, pl.span.end, pl.span.total })
    else
        log.info("[dist] worker: {s}, half of every expert (tensor)\n", .{name});

    const fail = struct {
        fn send(l: *const Link, code: i32) void {
            sendMsg(l, .{ MAGIC, code, 0, 0, 0, 0, 0, 0 }, 0) catch {};
        }
    };
    const dir = (try findModel(io, allocator, roots.items, name)) orelse {
        log.err("[dist] worker: no model named {s} in {s}\n", .{ name, roots.items[0] });
        fail.send(&link, NOT_FOUND);
        return error.DistModelNotFound;
    };
    defer allocator.free(dir);
    var config = try model_mod.parseConfig(io, allocator, dir);
    defer config.deinit(allocator);
    if (!pl.matches(&config)) {
        log.err("[dist] worker: {s} here has a different geometry than the host's\n", .{dir});
        fail.send(&link, MISMATCH);
        return error.DistModelMismatch;
    }
    var xfm = load(io, allocator, dir, &config, pl.mode, pl.span) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        fail.send(&link, LOAD_FAILED);
        return err;
    };
    defer xfm.t.deinit();
    defer xfm.w.deinit();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try sendMsg(&link, .{ MAGIC, 0, mib(active), 0, 0, 0, 0, 0 }, 0);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});

    const t = &xfm.t;
    var ctx = t.defaultCtx();
    var forwards: u64 = 0;
    while (true) {
        const m = try recvMsg(&link, 0);
        switch (m[0]) {
            OP_STOP => break,
            OP_FORWARD => {
                const offset: usize = @as(usize, @intCast(m[3])) | (@as(usize, @intCast(m[4])) << 31);
                if (offset == 0) {
                    try t.resetCache();
                } else if (offset != t.moe_seq_offset) {
                    log.err("[dist] forward at {d} but this worker holds {d} tokens\n", .{ offset, t.moe_seq_offset });
                    return error.DistPeerDiverged;
                }
                const shape = [_]c_int{ m[1], m[2], m[5] };
                const h = try queuedRecv(&link, &shape, @enumFromInt(m[6]), 0);
                defer _ = mlx.mlx_array_free(h);
                const raw = try t.forwardQwen4Stream(&ctx, h);
                defer _ = mlx.mlx_array_free(raw);
                // Rank 0 receives exactly the bytes it sent: JACCL sizes its buffers from the byte count.
                if (!std.mem.eql(c_int, mlx.getShape(raw), mlx.getShape(h))) return error.DistBadMessage;
                var out = raw;
                if (mlx.mlx_array_dtype(raw) != mlx.mlx_array_dtype(h)) {
                    out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(h), t.s));
                }
                defer if (out.ctx != raw.ctx) {
                    _ = mlx.mlx_array_free(out);
                };
                const sent = try link.send(out, 0);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                forwards += 1;
                if (m[1] * m[2] > 1 or forwards % 256 == 0) log.info("[dist] worker: forward {d}: {d}x{d} rows, now at {d} tokens\n", .{ forwards, m[1], m[2], t.moe_seq_offset });
            },
            OP_MOE => {
                const x = try queuedRecv(&link, &.{ m[1], m[2], m[3] }, @enumFromInt(m[5]), 0);
                defer _ = mlx.mlx_array_free(x);
                const logits = try queuedRecv(&link, &.{ m[1], m[2], m[4] }, @enumFromInt(m[7]), 0);
                defer _ = mlx.mlx_array_free(logits);
                const raw = try t.moeExpertsRouted(x, @intCast(m[6]), logits);
                defer _ = mlx.mlx_array_free(raw);
                if (!std.mem.eql(c_int, mlx.getShape(raw), mlx.getShape(x))) return error.DistBadMessage;
                var out = raw;
                if (mlx.mlx_array_dtype(raw) != mlx.mlx_array_dtype(x)) {
                    out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(x), t.s));
                }
                defer if (out.ctx != raw.ctx) {
                    _ = mlx.mlx_array_free(out);
                };
                const sent = try link.send(out, 0);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                forwards += 1;
            },
            else => return error.DistBadMessage,
        }
    }
    log.info("[dist] worker: stopped by rank 0 after {d} forwards\n", .{forwards});
}

const Loaded = struct { w: model_mod.Weights, t: transformer_mod.Transformer };

fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, config: *model_mod.ModelConfig, mode: Mode, span: LayerSpan) !Loaded {
    if (!config.isQwen4()) return error.DistArchUnsupported;
    switch (mode) {
        .pipeline => try config.applyLayerSpan(span),
        .tensor => try config.applyShard(.{ .rank = 1, .size = 2 }),
    }
    var w = try model_mod.loadModelWeights(io, allocator, dir, config, false);
    errdefer w.deinit();
    var t = try transformer_mod.Transformer.init(io, allocator, config.*, &w);
    errdefer t.deinit();
    _ = mlx.applyWiredPolicy();
    if (t.moe_layers != null) t.compileMoeRouting();
    if (config.linear_num_key_heads > 0) t.compileGdnGate();
    return .{ .w = w, .t = t };
}

fn evalOne(a: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

/// `mlx-serve dist-probe --rank R --coordinator IP:PORT --device rdma_enX`:
/// joins the group and measures round trips at the sizes inference moves
/// (one decode row of a hyper-connection stream, a prefill chunk) and all_sum.
pub fn cmdProbe(io: std.Io, args: []const []const u8) !void {
    const cfg = try parseArgs(args);
    var link = try Link.open(io, cfg);
    defer link.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const other: u32 = if (link.rank == 0) 1 else 0;

    const sizes = [_]c_int{ 1, 5120, 10240, 262144, 6553600 }; // f32 elements: 4 B .. 25 MB
    for (sizes) |n| {
        const shape = [_]c_int{n};
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&x, &shape, 1, .float32, s));
        defer _ = mlx.mlx_array_free(x);
        try evalOne(x);
        const iters: u32 = if (n > 1_000_000) 20 else 200;
        var best: u64 = std.math.maxInt(u64);
        var total: u64 = 0;
        for (0..iters + 5) |i| {
            var sw = io_util.Stopwatch.init(io);
            if (link.rank == 0) {
                const sent = try link.send(x, other);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                const back = try link.recv(&shape, .float32, other);
                defer _ = mlx.mlx_array_free(back);
                try evalOne(back);
            } else {
                const got = try link.recv(&shape, .float32, other);
                defer _ = mlx.mlx_array_free(got);
                try evalOne(got);
                const sent = try link.send(got, other);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
            }
            const ns = sw.read();
            if (i >= 5) {
                total += ns;
                best = @min(best, ns);
            }
        }
        const bytes: f64 = @floatFromInt(@as(u64, @intCast(n)) * 4);
        const mean_us = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        log.info("[dist-probe] round trip {d:>9} B: mean {d:>9.1} us, best {d:>9.1} us, {d:>6.2} GB/s each way\n", .{ n * 4, mean_us, @as(f64, @floatFromInt(best)) / 1000.0, 2.0 * bytes / (mean_us * 1000.0) });
    }
    for ([_]c_int{ 10240, 262144 }) |n| {
        const shape = [_]c_int{n};
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&x, &shape, 1, .bfloat16, s));
        defer _ = mlx.mlx_array_free(x);
        try evalOne(x);
        var total: u64 = 0;
        const iters: u32 = 200;
        var v: f32 = 0;
        for (0..iters + 5) |i| {
            var sw = io_util.Stopwatch.init(io);
            const r = try link.allSum(x);
            defer _ = mlx.mlx_array_free(r);
            try evalOne(r);
            if (i >= 5) total += sw.read();
            if (i == 0) {
                var first = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(first);
                var f32a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(f32a);
                try mlx.check(mlx.mlx_astype(&f32a, r, .float32, s));
                try mlx.check(mlx.mlx_sum(&first, f32a, false, s));
                try mlx.check(mlx.mlx_array_item_float32(&v, first));
            }
        }
        log.info("[dist-probe] all_sum bf16 {d:>9} B: mean {d:>9.1} us (sum check {d} == {d})\n", .{ n * 2, @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1000.0, v, n * @as(c_int, @intCast(link.size)) });
    }
    log.info("[dist-probe] done\n", .{});
}

pub fn parseArgs(args: []const []const u8) !Config {
    var rank: ?u32 = null;
    var coordinator: ?[]const u8 = null;
    var device: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (i + 1 >= args.len) {
            log.err("[dist] {s} needs a value\n", .{a});
            return error.InvalidArgs;
        }
        const v = args[i + 1];
        if (std.mem.eql(u8, a, "--rank")) {
            rank = std.fmt.parseInt(u32, v, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--coordinator")) {
            coordinator = v;
        } else if (std.mem.eql(u8, a, "--device")) {
            device = v;
        } else {
            log.err("[dist] unknown option {s}\n", .{a});
            return error.InvalidArgs;
        }
        i += 1;
    }
    if (rank == null or coordinator == null or device == null) {
        log.err("usage: mlx-serve dist-probe --rank N --coordinator IP:PORT --device rdma_enX\n", .{});
        return error.InvalidArgs;
    }
    return .{ .rank = rank.?, .coordinator = coordinator.?, .device = device.? };
}

test "deviceMatrixJson: each row names the local device, the diagonal is null" {
    var buf: [256]u8 = undefined;
    const j = try deviceMatrixJson(.{ .rank = 1, .coordinator = "x:1", .device = "rdma_en1" }, &buf);
    try std.testing.expectEqualStrings("[[null,\"rdma_en1\"],[\"rdma_en1\",null]]", j);
}

test "LayerSpan.remap: a later rank renumbers its trunk layers and drops the head-only towers" {
    var buf: [128]u8 = undefined;
    const tail: LayerSpan = .{ .start = 32, .end = 48, .total = 48 };
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.gate.weight", tail.remap("model.language_model.layers.32.mlp.gate.weight", &buf).?);
    try std.testing.expectEqualStrings("language_model.model.layers.15.self_attn.q_proj.weight", tail.remap("language_model.model.layers.47.self_attn.q_proj.weight", &buf).?);
    try std.testing.expect(tail.remap("model.language_model.layers.31.mlp.gate.weight", &buf) == null);
    try std.testing.expect(tail.remap("mtp.layers.0.attn_hyper_connection.hc_norm.weight", &buf) == null);
    try std.testing.expect(tail.remap("model.visual.blocks.3.attn.qkv.weight", &buf) == null);
    try std.testing.expectEqualStrings("lm_head.weight", tail.remap("lm_head.weight", &buf).?);

    const head: LayerSpan = .{ .start = 0, .end = 32, .total = 48 };
    try std.testing.expectEqualStrings("model.language_model.layers.31.mlp.gate.weight", head.remap("model.language_model.layers.31.mlp.gate.weight", &buf).?);
    try std.testing.expect(head.remap("model.language_model.layers.32.mlp.gate.weight", &buf) == null);
    try std.testing.expectEqualStrings("mtp.layers.0.attn_hyper_connection.hc_norm.weight", head.remap("mtp.layers.0.attn_hyper_connection.hc_norm.weight", &buf).?);
}

test "chooseSplit: a 2:1 memory split puts two thirds of the layers on rank 0, on a group boundary" {
    try std.testing.expectEqual(@as(u32, 32), chooseSplit(48, 4, 240_000, 120_000));
    try std.testing.expectEqual(@as(u32, 24), chooseSplit(48, 4, 100, 100));
    try std.testing.expectEqual(@as(u32, 44), chooseSplit(48, 4, 1000, 1));
    try std.testing.expectEqual(@as(u32, 4), chooseSplit(48, 4, 1, 1000));
}

test "Shard: gate/up keep a band of rows per expert, down a band of columns" {
    try std.testing.expect(Shard.expertAxis("language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight").? == .rows);
    try std.testing.expect(Shard.expertAxis("language_model.model.layers.3.mlp.shared_expert.down_proj.scales").? == .cols);
    try std.testing.expect(Shard.expertAxis("language_model.model.layers.3.mlp.switch_mlp.gate_proj.expert_scale") == null);
    try std.testing.expect(Shard.expertAxis("language_model.model.layers.3.mlp.shared_expert_gate.weight") == null);
    try std.testing.expect(Shard.expertAxis("mtp.layers.0.mlp.switch_mlp.up_proj.weight") == null);
    // [2 experts, 4 rows, 2 bytes]: rank 1 of 2 keeps rows 2..3 of each expert.
    const buf = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17 };
    var out: [8]u8 = undefined;
    (Shard{ .rank = 1, .size = 2 }).slice(&out, &buf, 2, 4, 2);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 14, 15, 16, 17 }, &out);
}

test "clusterJson: a pipeline split names both ranks' layers" {
    var p: Peer = .{ .link = undefined, .mode = .pipeline, .model = "/m/org/Flash-FP4", .split = null, .worker = .{ .avail_mib = 100, .total_mib = 128 }, .span = .{ .start = 0, .end = 16, .total = 48 }, .planned = true, .ready = true, .exchanges = 3 };
    const j = try clusterJson(std.testing.allocator, &p);
    defer std.testing.allocator.free(j);
    try std.testing.expectEqualStrings(",\"cluster\":{\"mode\":\"pipeline\",\"model\":\"Flash-FP4\",\"joined\":true,\"planned\":true,\"worker_ready\":true,\"layers\":48,\"here\":[0,16],\"worker\":[16,48],\"expert_shards\":0,\"worker_free_mib\":100,\"worker_total_mib\":128,\"exchanges\":3}", j);
}
