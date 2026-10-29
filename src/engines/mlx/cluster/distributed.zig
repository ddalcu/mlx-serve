//! Multi-Mac inference over MLX's distributed group (JACCL = RDMA over
//! Thunderbolt 5, macOS 26.2+). Rank 0 is the server; every other rank is a
//! worker that executes what rank 0 tells it, in lockstep.
const std = @import("std");
const env = @import("../../../env.zig");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const crt = @import("../../../sdk/util/crt.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const status = @import("../../../host/status.zig");
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const h3_split = @import("../../../plugins/minimax_h3/cluster.zig");
const wire = @import("../../../sdk/util/wire.zig");
const lan = @import("../../../host/lan/lan.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");

pub const Mode = enum {
    /// Each rank owns a contiguous range of layers; the hidden stream hops at the seams.
    pipeline,
    /// Every rank runs every layer on its slice of each MoE expert; one all_sum per MoE layer.
    tensor,
    /// A video DiT whole on every rank, each step's rows split (`h3_split`); the group's
    /// model decides it, never the flag.
    rows,

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

pub const LayerSpan = @import("../../../sdk/weights/shard.zig").LayerSpan;

pub const Shard = @import("../../../sdk/weights/shard.zig").Shard;

pub const headOnly = @import("../../../sdk/weights/shard.zig").headOnly;

pub const Link = struct {
    group: mlx.mlx_distributed_group,
    rank: u32,
    size: u32,
    bell: Bell = .{},

    /// Joins the group. Blocks until every rank has connected to the coordinator.
    pub fn open(io: std.Io, cfg: Config) !Link {
        if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable;
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
        _ = env.setForeign("MLX_RANK", try std.fmt.bufPrintSentinel(&rank_buf, "{d}", .{cfg.rank}, 0));
        _ = env.setForeign("MLX_JACCL_COORDINATOR", try std.fmt.bufPrintSentinel(&coord_buf, "{s}", .{cfg.coordinator}, 0));
        _ = env.setForeign("MLX_IBV_DEVICES", path);
        log.info("[dist] joining as rank {d}/{d} via {s}, coordinator {s}\n", .{ cfg.rank, cfg.size, cfg.device, cfg.coordinator });
        // Rank 0 listens for the bell before the group forms, so rank 1 finds it once it does.
        const coord = try parseCoordinator(cfg.coordinator);
        const listener: std.c.fd_t = if (cfg.rank == 0) Bell.listen(coord) else Bell.none;
        defer if (listener >= 0) {
            _ = std.c.close(listener);
        };
        var g = mlx.mlx_distributed_group_new();
        mlx.check(mlx.mlx_distributed_init(&g, true, "jaccl")) catch {
            var eb: [512]u8 = undefined;
            log.err("[dist] JACCL init failed: {s}\n", .{mlx.takeError(&eb) orelse "unknown"});
            return error.RdmaInitFailed;
        };
        var link: Link = .{ .group = g, .rank = @intCast(mlx.mlx_distributed_group_rank(g)), .size = @intCast(mlx.mlx_distributed_group_size(g)) };
        if (link.rank != cfg.rank or link.size != cfg.size) return error.RdmaGroupMismatch;
        link.bell = if (link.rank == 0) Bell.accept(listener) else Bell.connect(coord);
        if (!link.bell.open()) log.warn("[dist] no bell connection: the worker waits on the RDMA link, one core busy\n", .{});
        log.info("[dist] joined: rank {d} of {d}\n", .{ link.rank, link.size });
        return link;
    }

    pub fn deinit(self: *Link) void {
        if (self.bell.open()) _ = std.c.close(self.bell.fd);
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

    /// Every rank's `x` stacked on axis 0 in rank order; the shapes must match.
    pub fn allGather(self: *const Link, x: mlx.mlx_array) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_all_gather(&r, x, self.group, cs));
        return r;
    }
};

/// A plain TCP connection beside the RDMA link, on the coordinator's port + 1. Rank 0 writes
/// one byte before each control message and rank 1 sleeps on it before receiving one: JACCL
/// waits by polling the link in a loop, which held a core at 100% while the worker idled.
const Bell = struct {
    fd: std.c.fd_t = none,

    /// No connection. `fd_t` is a HANDLE on Windows, where a cluster never forms.
    const none: std.c.fd_t = if (builtin.os.tag == .windows) std.os.windows.INVALID_HANDLE_VALUE else -1;

    fn open(b: Bell) bool {
        return b.fd != none;
    }

    fn listen(coord: Coordinator) std.c.fd_t {
        const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (fd < 0) return none;
        const one: c_int = 1;
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, &one, @sizeOf(c_int));
        var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, coord.port +% 1), .addr = @bitCast(coord.ip4) };
        if (std.c.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0 or std.c.listen(fd, 1) != 0) {
            _ = std.c.close(fd);
            return none;
        }
        return fd;
    }

    /// Rank 1 connects right after the group forms; it waits up to 10 s.
    fn accept(listener: std.c.fd_t) Bell {
        if (listener == none) return .{};
        var fds = [_]std.posix.pollfd{.{ .fd = listener, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 10_000) catch return .{};
        if (ready == 0) return .{};
        const fd = std.c.accept(listener, null, null);
        return .{ .fd = if (fd < 0) none else noDelay(fd) };
    }

    fn connect(coord: Coordinator) Bell {
        const fd = lan.connectTimeout(coord.ip4, coord.port +% 1, 3000) catch return .{};
        return .{ .fd = noDelay(fd) };
    }

    fn noDelay(fd: std.c.fd_t) std.c.fd_t {
        const one: c_int = 1;
        _ = std.c.setsockopt(fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &one, @sizeOf(c_int));
        return fd;
    }

    fn writeAll(b: Bell, bytes: []const u8) !void {
        var off: usize = 0;
        while (off < bytes.len) {
            const n = std.c.write(b.fd, bytes.ptr + off, bytes.len - off);
            if (n <= 0) return error.DistPeerGone;
            off += @intCast(n);
        }
    }

    fn readAll(b: Bell, out: []u8) !void {
        var off: usize = 0;
        while (off < out.len) {
            const n = std.c.read(b.fd, out.ptr + off, out.len - off);
            if (n < 0 and std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            if (n <= 0) return error.DistPeerGone;
            off += @intCast(n);
        }
    }

    fn ring(b: Bell) !void {
        if (!b.open()) return;
        if (std.c.write(b.fd, &[_]u8{1}, 1) != 1) return error.DistPeerGone;
    }

    fn wait(b: Bell) !void {
        if (!b.open()) return;
        var x: [1]u8 = undefined;
        while (true) {
            const n = std.c.read(b.fd, &x, 1);
            if (n == 1) return;
            if (n < 0 and std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.DistPeerGone;
        }
    }
};

const Coordinator = struct { ip4: [4]u8, port: u16 };

fn parseCoordinator(text: []const u8) !Coordinator {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.DistBadCoordinator;
    var out: Coordinator = .{ .ip4 = undefined, .port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.DistBadCoordinator };
    var it = std.mem.splitScalar(u8, text[0..colon], '.');
    for (&out.ip4) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.DistBadCoordinator, 10) catch return error.DistBadCoordinator;
    if (it.next() != null) return error.DistBadCoordinator;
    return out;
}

// ── Wire protocol: fixed int32[8] control messages, then any payload tensor ──

pub const MAGIC = wire.MAGIC;
const VERSION: i32 = 5;
pub const Msg = wire.Msg;
const OP_FORWARD: i32 = 1;
pub const OP_STOP = wire.OP_STOP;
const OP_MOE: i32 = 3;
pub const OP_H3_JOB = wire.OP_H3_JOB;
pub const OP_H3_STEP = wire.OP_H3_STEP;
pub const OP_H3_END = wire.OP_H3_END;
const OP_DSV41: i32 = 7;
pub const OP_RELEASE = wire.OP_RELEASE;

/// `link` as a plugin's split job talks over it (`sdk/util/wire.zig`).
pub fn wireOf(link: *const Link) wire.Link {
    return .{ .ctx = link, .vt = &WireOps.vt };
}

const WireOps = struct {
    const vt: wire.Link.VTable = .{ .send_op = sendOp_, .recv_op = recvOp_, .send_now = sendNow_, .recv_now = recvNow_, .all_gather = allGather_ };

    fn of(ctx: *const anyopaque) *const Link {
        return @ptrCast(@alignCast(ctx));
    }
    fn sendOp_(ctx: *const anyopaque, m: Msg) anyerror!void {
        return sendOp(of(ctx), m);
    }
    fn recvOp_(ctx: *const anyopaque) anyerror!Msg {
        return recvOp(of(ctx));
    }
    fn sendNow_(ctx: *const anyopaque, x: mlx.mlx_array, dst: u32) anyerror!void {
        return sendNow(of(ctx), x, dst);
    }
    fn recvNow_(ctx: *const anyopaque, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) anyerror!mlx.mlx_array {
        return recvNow(of(ctx), shape, dtype, src);
    }
    fn allGather_(ctx: *const anyopaque, x: mlx.mlx_array) anyerror!mlx.mlx_array {
        return of(ctx).allGather(x);
    }
};

/// A control message to the other rank of a two-Mac group.
pub fn sendOp(link: *const Link, m: Msg) !void {
    return sendMsg(link, m, 1 - link.rank);
}

pub fn recvOp(link: *const Link) !Msg {
    return recvMsg(link, 1 - link.rank);
}

/// `x` sent and complete before this returns.
pub fn sendNow(link: *const Link, x: mlx.mlx_array, dst: u32) !void {
    const r = try link.send(x, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

pub fn recvNow(link: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
    return queuedRecv(link, shape, dtype, src);
}

fn sendMsg(link: *const Link, m: Msg, dst: u32) !void {
    if (link.rank == 0) try link.bell.ring();
    const shape = [_]c_int{8};
    const a = mlx.mlx_array_new_data(&m, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvMsg(link: *const Link, src: u32) !Msg {
    if (link.rank == 1) try link.bell.wait();
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

/// What a worker reports on joining: its memory, then its host name (`name_len` bytes).
const Hello = struct {
    avail_mib: i32,
    total_mib: i32,
    name_len: i32 = 0,

    fn here(name_len: usize) Hello {
        return .{ .avail_mib = mib(status.getAvailableMemBytes()), .total_mib = mib(status.getTotalMemBytes()), .name_len = @intCast(name_len) };
    }
    fn msg(h: Hello) Msg {
        return .{ MAGIC, VERSION, h.avail_mib, h.total_mib, h.name_len, 0, 0, 0 };
    }
    fn parse(m: Msg) !Hello {
        if (m[0] != MAGIC or m[1] != VERSION) return error.DistProtocolMismatch;
        if (m[4] < 0 or m[4] > 64) return error.DistBadMessage;
        return .{ .avail_mib = m[2], .total_mib = m[3], .name_len = m[4] };
    }
};

/// What rank 0 asks a worker to load: the model by folder name, then its `modelIdentity`,
/// checked against the worker's own copy, and its part of it.
const Plan = struct {
    mode: Mode,
    span: LayerSpan,
    hidden: i32,
    experts: i32,

    fn msg(p: Plan, name_len: usize) Msg {
        return .{ MAGIC, @backingInt(p.mode), @intCast(p.span.start), @intCast(p.span.end), @intCast(p.span.total), @intCast(name_len), p.hidden, p.experts };
    }
    fn parse(m: Msg) !Plan {
        if (m[0] != MAGIC) return error.DistProtocolMismatch;
        return .{ .mode = @fromBackingInt(@intCast(m[1])), .span = .{ .start = @intCast(m[2]), .end = @intCast(m[3]), .total = @intCast(m[4]) }, .hidden = m[6], .experts = m[7] };
    }
    fn matches(p: Plan, cfg: *const model_mod.ModelConfig) bool {
        return cfg.num_hidden_layers == p.span.total and cfg.hidden_size == p.hidden and cfg.num_experts == p.experts;
    }
    /// Sends the plan for `model_dir`; returns that copy's `modelIdentity`.
    fn send(p: Plan, link: *const Link, model_dir: []const u8) !u64 {
        const g = cluster().group.?;
        const id = try modelIdentity(g.io, g.allocator, model_dir);
        const name = std.fs.path.basename(model_dir);
        try sendMsg(link, p.msg(name.len), 1);
        try sendBytes(link, name, 1);
        try sendMsg(link, .{ MAGIC, @bitCast(@as(u32, @truncate(id))), @bitCast(@as(u32, @truncate(id >> 32))), 0, 0, 0, 0, 0 }, 1);
        return id;
    }
};

/// One copy of a model: its `config.json` and the path and size of every weight file under
/// it, so two folders of one name in another quantization or revision never pair.
fn modelIdentity(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var dir = try std.Io.Dir.cwd().openDir(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    const cfg = try dir.readFileAlloc(io, "config.json", allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(cfg);
    var id = std.hash.Wyhash.hash(0, cfg);
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |e| {
        if ((e.kind != .file and e.kind != .sym_link) or !(std.mem.endsWith(u8, e.basename, ".safetensors") or std.mem.endsWith(u8, e.basename, ".bin"))) continue;
        const st = try e.dir.statFile(io, e.basename, .{});
        // Summed, so the walk's order does not matter.
        id +%= std.hash.Wyhash.hash(st.size, e.path);
    }
    return id;
}

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
    /// The `modelIdentity` of the copy the plan was made for.
    plan_id: u64 = 0,
    /// DeepSeek-V4.1: the worker's layers in the middle or tail of the trunk.
    remote: ?LayerSpan = null,
    ready: bool = false,
    /// The worker refused or failed its load: it has left, and nothing waits on it again.
    failed: bool = false,
    /// What the worker's load left resident (its READY message).
    worker_resident_mib: u32 = 0,
    /// Exchanges with the worker since the group formed (`/props` cluster).
    exchanges: u64 = 0,
    /// Tokens the worker's caches hold. Every forward must continue them or start over at 0.
    peer_pos: usize = 0,
    /// The worker's host name, as its LAN sharing names it.
    worker_name_buf: [64]u8 = undefined,
    worker_name_len: usize = 0,
    /// Rows mode: each Mac's time for one calibration pass, and the link's measured rate.
    host_calib_us: u64 = 0,
    worker_calib_us: u64 = 0,
    link_bytes_per_us: f64 = 0,
    /// `link` as the rows job talks over it (`rowsPair`).
    split_link: wire.Link = undefined,

    pub fn workerName(self: *const Peer) []const u8 {
        return self.worker_name_buf[0..self.worker_name_len];
    }

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
        try self.sendOrdered(.{ OP_FORWARD, sh[0], sh[1], @intCast(offset & 0x7fffffff), @intCast(offset >> 31), sh[2], @backingInt(mlx.mlx_array_dtype(h)), 0 }, &.{h});
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
        try self.sendOrdered(.{ OP_MOE, xs[0], xs[1], xs[2], ls[ls.len - 1], @backingInt(mlx.mlx_array_dtype(x)), @intCast(layer), @backingInt(mlx.mlx_array_dtype(logits)) }, &.{ x, logits });
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
        try self.link.bell.ring();
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

    /// The worker holds its part of the copy the plan was made for: another folder of the same
    /// name (another quantization or revision) would pair this Mac's part with the wrong weights.
    fn samePlannedCopy(self: *const Peer, model_dir: []const u8) !void {
        const g = cluster().group.?;
        if (try modelIdentity(g.io, g.allocator, model_dir) == self.plan_id) return;
        log.err("[dist] {s} is not the copy the worker loaded; restart the server to pair with it\n", .{model_dir});
        return error.DistModelMismatch;
    }

    fn awaitReady(self: *Peer) !void {
        if (self.failed) return error.DistWorkerLoadFailed;
        const m = try recvMsg(&self.link, 1);
        if (m[0] != MAGIC or m[1] != 0) {
            self.failed = true;
            switch (m[1]) {
                NOT_FOUND => log.err("[dist] the worker has no model named {s} in its model folders\n", .{self.model}),
                MISMATCH => log.err("[dist] the worker's copy of {s} is not the same model (quantization, revision or geometry)\n", .{self.model}),
                else => log.err("[dist] the worker failed to load its part of {s}\n", .{self.model}),
            }
            return error.DistWorkerLoadFailed;
        }
        self.worker_resident_mib = @intCast(@max(m[2], 0));
        switch (self.mode) {
            .pipeline => if (self.remote) |r|
                log.info("[dist] worker ready: layers {d}..{d} resident, {d} MiB\n", .{ r.start, r.end, m[2] })
            else
                log.info("[dist] worker ready: layers {d}..{d} resident, {d} MiB\n", .{ self.span.end, self.span.total, m[2] }),
            .tensor => log.info("[dist] worker ready: its half of every expert resident, {d} MiB\n", .{m[2]}),
            .rows => {
                self.worker_calib_us = @intCast(m[3]);
                log.info("[dist] worker ready: the whole DiT resident, {d} MiB\n", .{m[2]});
            },
        }
        self.ready = true;
    }
};

/// A receive completed on the host before anything reads it. A lazy one lets a GPU command
/// buffer wait on it, and MLX signals a buffer's events only when the whole buffer ends: the
/// buffer that computes the payload can then also hold the wait for the reply to it, which
/// deadlocks until Metal's watchdog kills it. Done here, the receive also pairs with the
/// peer's send even when nobody reads it (a warmup that drops its output).
fn queuedRecv(link: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
    _ = cluster().recvs.fetchAdd(1, .acq_rel);
    defer _ = cluster().recvs.fetchSub(1, .acq_rel);
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

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ClusterState = struct {
    group: ?Group = null,
    joined: std.atomic.Value(bool) = .init(false),
    joined_link: Link = undefined,
    last_job_buf: [160]u8 = undefined,
    last_job_len: usize = 0,
    /// The inference thread writes the note while `/props` reads it.
    last_job_mu: std.atomic.Mutex = .unlocked,
    last_split: [2]u32 = .{ 0, 0 },
    peer_slot: ?Peer = null,
    /// Round trips since the last timing line, and their totals (µs): the whole exchange and the
    /// worker's share of it.
    timing: struct { n: u32 = 0, round_us: u64 = 0, worker_us: u64 = 0 } = .{},
    /// What the worker said about the group's model when it joined: 0 has it, else NOT_FOUND or
    /// MISMATCH (-1 = not asked: no bell connection).
    worker_copy: std.atomic.Value(i32) = .init(-1),
    /// The worker's compute time for its last step (µs), sent over the bell after each one.
    worker_us: std.atomic.Value(u32) = .init(0),
    /// Receives from the worker in progress on the inference thread (`queuedRecv`).
    recvs: std.atomic.Value(u32) = .init(0),
};
pub const cluster_shared = @import("../../../sdk/util/shared.zig").Shared(ClusterState, "cluster_state", .{});
inline fn cluster() *ClusterState {
    return cluster_shared.get();
}

/// The group this server leads (`--dist`): set at startup, while a thread waits
/// for the worker to join; the handshake runs when the group's model loads.
const Group = struct { io: std.Io, allocator: std.mem.Allocator, mode: Mode, model: []const u8, split: ?u32 };
/// The worker's bell connection closed: it left, and JACCL forms a group once per process,
/// so this server must restart to pair again (the app does it, `/props` cluster "worker_gone").
var worker_gone = std.atomic.Value(bool).init(false);

/// The worker's free memory as it answered on joining (MiB), for the load bill.
var worker_free_mib = std.atomic.Value(u32).init(0);

/// The `/props` "cluster" object (with its leading comma), empty when this server is
/// not rank 0 of a group: what the app draws the split from.
pub fn propsJson(allocator: std.mem.Allocator) ![]u8 {
    if (peer()) |p| return clusterJson(allocator, p);
    const g = cluster().group orelse return allocator.dupe(u8, "");
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":{f},\"joined\":{},\"planned\":false,\"worker_ready\":false,\"worker_model\":\"{s}\",\"worker_gone\":{}}}", .{
        @tagName(g.mode), std.json.fmt(std.fs.path.basename(g.model), .{}), cluster().joined.load(.acquire), workerCopyName(), worker_gone.load(.acquire),
    });
}

/// The `/v1/models` "machine" fields of a Thunderbolt pair this server leads (with a leading
/// comma), empty when it leads none: its model and whether the other Mac has joined, so that
/// Mac's app can offer to join it.
pub fn pairFields(allocator: std.mem.Allocator) ![]u8 {
    const g = cluster().group orelse return allocator.dupe(u8, "");
    return pairFieldsFor(allocator, std.fs.path.basename(g.model), cluster().joined.load(.acquire));
}

fn pairFieldsFor(allocator: std.mem.Allocator, model: []const u8, has_joined: bool) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"leads\":{f},\"joined\":{}", .{ std.json.fmt(model, .{}), has_joined });
}

fn clusterJson(allocator: std.mem.Allocator, p: *const Peer) ![]u8 {
    const sp = p.span;
    while (!cluster().last_job_mu.tryLock()) std.atomic.spinLoopHint();
    defer cluster().last_job_mu.unlock();
    // Rows mode: the worker's speed relative to this Mac and the link, once measured.
    const speed: f64 = if (p.worker_calib_us > 0 and p.host_calib_us > 0) @as(f64, @floatFromInt(p.host_calib_us)) / @as(f64, @floatFromInt(p.worker_calib_us)) else 0;
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":{f},\"joined\":{},\"planned\":{},\"worker_ready\":{},\"layers\":{d},\"here\":[{d},{d}],\"worker\":[{d},{d}],\"expert_shards\":{d},\"worker_free_mib\":{d},\"worker_total_mib\":{d},\"exchanges\":{d},\"worker_name\":{f},\"worker_speed\":{d:.2},\"link_gbps\":{d:.1},\"last_job\":{f},\"split_rows\":[{d},{d}],\"worker_model\":\"{s}\",\"worker_gone\":{}}}", .{
        @tagName(p.mode),                       std.json.fmt(std.fs.path.basename(p.model), .{}),                     cluster().joined.load(.acquire),  p.planned,               p.ready,            sp.total,                   sp.start,                          sp.end,
        if (p.mode == .pipeline) sp.end else 0, sp.total,                                                             if (p.shard) |sh| sh.size else 0, p.worker.avail_mib,      p.worker.total_mib, p.exchanges,                std.json.fmt(p.workerName(), .{}), speed,
        p.link_bytes_per_us / 1000,             std.json.fmt(cluster().last_job_buf[0..cluster().last_job_len], .{}), cluster().last_split[0],          cluster().last_split[1], workerCopyName(),   worker_gone.load(.acquire),
    });
}

/// `/props` cluster "worker_model": whether the worker has the group's model.
fn workerCopyName() []const u8 {
    return switch (cluster().worker_copy.load(.acquire)) {
        0 => "ok",
        NOT_FOUND => "missing",
        MISMATCH => "different",
        else => "unknown",
    };
}

/// What the last video job did with the worker (`/props` cluster "last_job", and its rows
/// on each Mac, zeros when it ran here alone).
fn noteJob(split: [2]u32, line: []const u8) void {
    while (!cluster().last_job_mu.tryLock()) std.atomic.spinLoopHint();
    defer cluster().last_job_mu.unlock();
    cluster().last_split = split;
    const n = if (line.len <= cluster().last_job_buf.len) line.len else 0;
    @memcpy(cluster().last_job_buf[0..n], line[0..n]);
    cluster().last_job_len = n;
}

/// Rank 0 at shutdown: releases the worker. A peer that vanishes mid-protocol crashes the
/// other side inside Apple's RDMA library, so the group always ends with this message.
pub fn shutdown() void {
    const p = peer() orelse return;
    if (!p.ready) return;
    // The stopped worker posts no more receives: a later message (the model's unload) would wait forever.
    p.ready = false;
    sendMsg(&p.link, .{ OP_STOP, 0, 0, 0, 0, 0, 0, 0 }, 1) catch |err| {
        var eb: [512]u8 = undefined;
        log.warn("[dist] the stop did not reach the worker ({s}: {s})\n", .{ @errorName(err), mlx.takeError(&eb) orelse "" });
        return;
    };
    log.info("[dist] worker released\n", .{});
}

/// Rank 0, on the inference thread, when the pair's model unloads here: the worker frees its
/// part too and waits for the next plan, which the model's next load sends.
pub fn release() void {
    const p = peer() orelse return;
    if (!p.ready) return;
    sendMsg(&p.link, .{ OP_RELEASE, 0, 0, 0, 0, 0, 0, 0 }, 1) catch return;
    p.planned = false;
    p.ready = false;
    p.remote = null;
    p.shard = null;
    p.peer_pos = 0;
    log.info("[dist] {s} unloaded: the worker released its part\n", .{std.fs.path.basename(p.model)});
}

/// Rank 0's peer, once `connect` has run.
pub fn peer() ?*Peer {
    return if (cluster().peer_slot) |*p| p else null;
}

/// What this rank holds of `model_dir`'s model when it is the group's model: on the first
/// load the split is chosen and the worker told to load its part; reloads reuse it.
pub const Part = union(enum) { span: LayerSpan, shard: Shard, remote: LayerSpan };

pub fn planFor(model_dir: []const u8, config: *const model_mod.ModelConfig) !?Part {
    if (!serves(model_dir)) return null;
    const p = try join();
    if (p.planned) try p.samePlannedCopy(model_dir) else try plan(p, config, model_dir);
    if (p.remote) |r| return .{ .remote = r };
    return if (p.shard) |sh| .{ .shard = sh } else .{ .span = p.span };
}

/// What this Mac holds of `whole` resident bytes of `model_dir` when its pair serves it: its run
/// of layers, chosen as the plan chooses it (or the plan itself once made); `whole` otherwise.
pub fn hostShare(model_dir: []const u8, config: *const model_mod.ModelConfig, whole: u64) u64 {
    if (!holds(model_dir)) return whole;
    const g = cluster().group.?;
    const n: u64 = config.num_hidden_layers;
    if (n == 0) return whole;
    if (peer()) |p| if (p.planned) return if (p.mode == .tensor) whole / 2 + whole / 10 else whole * p.span.len() / n;
    return shareFor(g, config, whole, status.getAvailableMemBytes() >> 20, worker_free_mib.load(.acquire));
}

fn shareFor(g: Group, config: *const model_mod.ModelConfig, whole: u64, mine: u64, theirs: u64) u64 {
    const n: u64 = config.num_hidden_layers;
    if (theirs == 0) return whole;
    if (config.isDsv41()) {
        const r = dsv41_mod.pairRange(config, mine, theirs) orelse return whole;
        return whole * (n - (r[1] - r[0])) / n;
    }
    if (!config.isQwen4()) return whole;
    return switch (g.mode) {
        .tensor => whole / 2 + whole / 10,
        .rows => whole,
        .pipeline => whole * (g.split orelse chooseSplit(@intCast(n), config.full_attention_interval, mine, theirs)) / n,
    };
}

/// Whether a Thunderbolt pair holds `model_dir`: the worker joined, so no other load unloads it
/// to make room (an explicit unload still does, on both Macs: `release`).
pub fn holds(model_dir: []const u8) bool {
    return cluster().joined.load(.acquire) and serves(model_dir);
}

/// Whether `model_dir` is the model this server's group serves.
pub fn serves(model_dir: []const u8) bool {
    if (comptime builtin.os.tag != .macos) return false; // a cluster exists only on macOS
    const g = cluster().group orelse return false;
    return samePath(g.model, model_dir);
}

/// One folder however it is spelled (symlinks, `..`, a trailing slash); a path that does not
/// resolve matches only itself. A folder elsewhere with the same name is another copy.
fn samePath(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    var za: [std.fs.max_path_bytes]u8 = undefined;
    var zb: [std.fs.max_path_bytes]u8 = undefined;
    var ra: [std.fs.max_path_bytes]u8 = undefined;
    var rb: [std.fs.max_path_bytes]u8 = undefined;
    if (a.len >= za.len or b.len >= zb.len) return false;
    @memcpy(za[0..a.len], a);
    za[a.len] = 0;
    @memcpy(zb[0..b.len], b);
    zb[b.len] = 0;
    const pa = std.c.realpath(@ptrCast(&za), &ra) orelse return false;
    const pb = std.c.realpath(@ptrCast(&zb), &rb) orelse return false;
    return std.mem.eql(u8, std.mem.span(pa), std.mem.span(pb));
}

/// `planRows` as a split job reads it (`h3_split.Pair`).
fn rowsPair(model_dir: []const u8, hidden: i32, blocks: i32) anyerror!h3_split.Pair {
    const p = try planRows(model_dir, hidden, blocks);
    p.split_link = wireOf(&p.link);
    return .{
        .link = &p.split_link,
        .host_calib_us = &p.host_calib_us,
        .link_bytes_per_us = &p.link_bytes_per_us,
        .exchanges = &p.exchanges,
        .worker_calib_us = p.worker_calib_us,
        .worker_free_mib = @as(u64, @intCast(@max(p.worker.avail_mib, 0))) -| p.worker_resident_mib,
    };
}

/// Rows mode: the worker told to load the group's video model whole, and ready.
fn planRows(model_dir: []const u8, hidden: i32, blocks: i32) !*Peer {
    const p = try join();
    if (!p.planned) {
        p.mode = .rows;
        const pl: Plan = .{ .mode = .rows, .span = .{ .start = 0, .end = @intCast(blocks), .total = @intCast(blocks) }, .hidden = hidden, .experts = 0 };
        p.plan_id = try pl.send(&p.link, model_dir);
        p.planned = true;
    } else try p.samePlannedCopy(model_dir);
    if (!p.ready) try p.awaitReady();
    return p;
}

/// The worker's hello, read once.
fn join() !*Peer {
    if (cluster().peer_slot == null) {
        const g = cluster().group.?;
        if (!cluster().joined.load(.acquire)) {
            log.err("[dist] {s} is served with a worker: start `mlx-serve worker` on the other Mac first\n", .{std.fs.path.basename(g.model)});
            return error.DistWorkerNotJoined;
        }
        switch (cluster().worker_copy.load(.acquire)) {
            NOT_FOUND, MISMATCH => {
                log.err("[dist] the worker does not have {s}: download the same model there, then join again\n", .{std.fs.path.basename(g.model)});
                return error.DistWorkerMissingModel;
            },
            else => {},
        }
        // The worker sends its hello only once rung: JACCL's send polls until this recv is posted.
        try cluster().joined_link.bell.ring();
        const worker = try Hello.parse(try recvMsg(&cluster().joined_link, 1));
        var p: Peer = .{ .link = cluster().joined_link, .mode = g.mode, .model = g.model, .split = g.split, .worker = worker };
        if (worker.name_len > 0) try recvBytes(&cluster().joined_link, p.worker_name_buf[0..@intCast(worker.name_len)], 1);
        p.worker_name_len = @intCast(worker.name_len);
        log.info("[dist] worker {s}: {d} MiB free of {d}\n", .{ p.workerName(), worker.avail_mib, worker.total_mib });
        cluster().peer_slot = p;
    }
    return &cluster().peer_slot.?;
}

fn plan(self: *Peer, config: *const model_mod.ModelConfig, model_dir: []const u8) !void {
    if (config.isDsv41()) return planDsv41(self, config, model_dir);
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
        .rows => return error.DistArchUnsupported,
        .pipeline => {
            const mine: u64 = status.getAvailableMemBytes() >> 20;
            const k = self.split orelse chooseSplit(n, config.full_attention_interval, mine, @intCast(self.worker.avail_mib));
            if (k == 0 or k >= n or (config.full_attention_interval != 0 and k % config.full_attention_interval != 0)) return error.DistSplitOffInterval;
            self.span = .{ .start = 0, .end = k, .total = n };
            pl.span = .{ .start = k, .end = n, .total = n };
            log.info("[dist] pipeline: layers 0..{d} of {s} here ({d} MiB free), {d}..{d} on the worker ({d} MiB free of {d})\n", .{ k, std.fs.path.basename(self.model), mine, k, n, self.worker.avail_mib, self.worker.total_mib });
        },
    }
    self.plan_id = try pl.send(&self.link, model_dir);
    self.planned = true;
}

/// DeepSeek-V4.1: the worker takes a run of layers its share of the memory holds, cut where
/// no layer reads another Mac's state (`deepseek_v41.pairRange`); the stream and mix cross twice.
fn planDsv41(self: *Peer, config: *const model_mod.ModelConfig, model_dir: []const u8) !void {
    if (self.mode != .pipeline) {
        log.err("[dist] {s}: DeepSeek-V4.1 splits by layers only (--dist pipeline)\n", .{self.model});
        return error.DistArchUnsupported;
    }
    const n = config.num_hidden_layers;
    const mine: u64 = status.getAvailableMemBytes() >> 20;
    const r = dsv41_mod.pairRange(config, mine, @intCast(self.worker.avail_mib)) orelse return error.DistSplitOffInterval;
    self.remote = .{ .start = @intCast(r[0]), .end = @intCast(r[1]), .total = n };
    self.span = .{ .start = 0, .end = @intCast(n - (r[1] - r[0])), .total = n };
    log.info("[dist] pipeline: layers {d}..{d} of {s} on the worker ({d} MiB free of {d}), the other {d} here ({d} MiB free)\n", .{ r[0], r[1], std.fs.path.basename(self.model), self.worker.avail_mib, self.worker.total_mib, n - (r[1] - r[0]), mine });
    const pl: Plan = .{ .mode = .pipeline, .span = self.remote.?, .hidden = @intCast(config.hidden_size), .experts = @intCast(config.num_experts) };
    self.plan_id = try pl.send(&self.link, model_dir);
    self.planned = true;
    dsv41_mod.clusterHooks().pair_exchange = &dsv41Exchange;
}

/// Host, DeepSeek-V4.1: one step's stream and mix through the worker's layers and back.
fn dsv41Exchange(x: mlx.mlx_array, pre: mlx.mlx_array, p0: usize, snap: bool, verify: bool) anyerror![2]mlx.mlx_array {
    const self = peer() orelse return error.DistWorkerNotJoined;
    if (!self.ready) try self.awaitReady();
    const xs = mlx.getShape(x);
    const t0 = io_util.nowUsMonotonic(cluster().group.?.io);
    defer noteRoundTrip(io_util.nowUsMonotonic(cluster().group.?.io) - t0);
    try self.sendOrdered(.{ OP_DSV41, xs[0], @intCast(p0 & 0x7fffffff), @intCast(p0 >> 31), @as(i32, @intFromBool(snap)) | @as(i32, @intFromBool(verify)) << 1, xs[1], @backingInt(mlx.mlx_array_dtype(x)), @backingInt(mlx.mlx_array_dtype(pre)) }, &.{ x, pre });
    self.exchanges += 1;
    const xo = try queuedRecv(&self.link, xs, mlx.mlx_array_dtype(x), 1);
    errdefer _ = mlx.mlx_array_free(xo);
    return .{ xo, try queuedRecv(&self.link, mlx.getShape(pre), mlx.mlx_array_dtype(pre), 1) };
}

/// Every 64 round trips: what one took, and how much of it the worker computed (the rest is the
/// link, both sends and the hand-off). The worker's time arrives over the bell, a step late.
fn noteRoundTrip(us: i64) void {
    cluster().timing.n += 1;
    cluster().timing.round_us += @intCast(@max(us, 0));
    cluster().timing.worker_us += cluster().worker_us.load(.acquire);
    if (cluster().timing.n < 64) return;
    const n: f64 = @floatFromInt(cluster().timing.n);
    const round = @as(f64, @floatFromInt(cluster().timing.round_us)) / n / 1000;
    const work = @as(f64, @floatFromInt(cluster().timing.worker_us)) / n / 1000;
    log.info("[dist] timing over {d} steps: round trip {d:.2} ms = worker {d:.2} ms + link {d:.2} ms\n", .{ cluster().timing.n, round, work, round - work });
    cluster().timing = .{};
}

/// Worker, DeepSeek-V4.1: its run of layers (`pl.span`), then one step per OP_DSV41 until
/// rank 0 stops it.
fn serveDsv41(io: std.Io, allocator: std.mem.Allocator, link: *const Link, dir: []const u8, config: *model_mod.ModelConfig, pl: Plan) !bool {
    config.dist_dsv41 = .{ .lo = pl.span.start, .hi = pl.span.end, .worker = true };
    var w = model_mod.loadModelWeights(io, allocator, dir, config, false) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        sendMsg(link, .{ MAGIC, LOAD_FAILED, 0, 0, 0, 0, 0, 0 }, 0) catch {};
        return err;
    };
    defer w.deinit();
    const binding = config.plugin orelse return error.NotPluginServed;
    const m = dsv41_mod.init(allocator, @import("../../../plugins/deepseek_v41/config.zig").of(&binding.parsed), &w, mlx.gpuStream(), .{ .pair = config.dist_dsv41 }) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        sendMsg(link, .{ MAGIC, LOAD_FAILED, 0, 0, 0, 0, 0, 0 }, 0) catch {};
        return err;
    };
    defer m.deinit();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try sendMsg(link, .{ MAGIC, 0, mib(active), 0, 0, 0, 0, 0 }, 0);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});
    const d: c_int = @intCast(config.hidden_size);
    var steps: u64 = 0;
    while (true) {
        const msg = try recvMsg(link, 0);
        switch (msg[0]) {
            OP_STOP => break,
            OP_RELEASE => return true,
            OP_DSV41 => {
                const p0: usize = @as(usize, @intCast(msg[2])) | (@as(usize, @intCast(msg[3])) << 31);
                const x = try queuedRecv(link, &.{ msg[1], msg[5], d }, @fromBackingInt(@intCast(msg[6])), 0);
                defer _ = mlx.mlx_array_free(x);
                const pre = try queuedRecv(link, &.{ msg[1], msg[5] }, @fromBackingInt(@intCast(msg[7])), 0);
                defer _ = mlx.mlx_array_free(pre);
                const t0 = io_util.nowUsMonotonic(io);
                const out = try dsv41_mod.workerStep(m, allocator, x, pre, p0, msg[4] & 1 != 0, msg[4] & 2 != 0);
                defer for (out) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                var took: [4]u8 = undefined;
                std.mem.writeInt(u32, &took, @intCast(std.math.clamp(io_util.nowUsMonotonic(io) - t0, 0, std.math.maxInt(u32))), .little);
                link.bell.writeAll(&took) catch {};
                try sendNow(link, out[0], 0);
                try sendNow(link, out[1], 0);
                steps += 1;
                if (msg[1] > 1 or steps % 256 == 0) log.info("[dist] worker: step {d}: {d} rows at {d}\n", .{ steps, msg[1], p0 });
            },
            else => return error.DistBadMessage,
        }
    }
    log.info("[dist] worker: stopped by rank 0 after {d} steps\n", .{steps});
    return false;
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
    if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable; // a cluster exists only on macOS
    cluster().group = .{ .io = io, .allocator = allocator, .mode = mode, .model = try allocator.dupe(u8, model), .split = split };
    h3_split.hooks().* = .{ .serves = &serves, .plan_rows = &rowsPair, .note_job = &noteJob };
    const t = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, waitForWorker, .{ io, cfg });
    t.detach();
}

fn waitForWorker(io: std.Io, cfg: Config) void {
    cluster().joined_link = Link.open(io, cfg) catch |err| {
        log.err("[dist] the cluster link did not open: {s}\n", .{@errorName(err)});
        return;
    };
    // Asked before `joined`: the inference thread's first ring must not land inside the question.
    askForModel(&cluster().joined_link) catch |err| {
        log.err("[dist] the worker did not answer about the model ({s}); is it an older mlx-serve?\n", .{@errorName(err)});
        return;
    };
    cluster().joined.store(true, .release);
    log.info("[dist] worker joined; it loads its part with {s}\n", .{std.fs.path.basename(cluster().group.?.model)});
    watchWorker(&cluster().joined_link);
}

/// Waits on the bell for the worker to leave: it never writes after its answer, so a read
/// returns only at its end.
fn watchWorker(link: *const Link) void {
    if (!link.bell.open()) return;
    // After its answer the worker writes only each step's compute time (µs, u32).
    var x: [4]u8 = undefined;
    while (link.bell.readAll(&x)) |_| {
        cluster().worker_us.store(std.mem.readInt(u32, &x, .little), .release);
    } else |_| {}
    worker_gone.store(true, .release);
    cluster().joined.store(false, .release);
    log.warn("[dist] the worker left; this server restarts to pair again\n", .{});
    // A receive from the worker, in progress or started later, never returns, and the inference thread with it: its
    // requests end by name and the process exits for the app (or a supervisor) to pair again.
    while (cluster().recvs.load(.acquire) == 0) crt.sleepMs(100);
    log.err("[dist] a step waits on the worker that left; its requests end and this server exits\n", .{});
    if (@import("../../../host/http/server.zig").launch().global_scheduler) |sch| @import("../scheduler.zig").failInFlight(sch, "DistWorkerLeft");
    crt.sleepMs(2000);
    std.process.exit(1);
}

/// Right after the worker joins, over the bell connection (no MLX off the inference thread):
/// the group's model by name and `modelIdentity`, answered with whether the worker has it.
fn askForModel(link: *const Link) !void {
    if (!link.bell.open()) return;
    const g = cluster().group.?;
    const name = std.fs.path.basename(g.model);
    const id = try modelIdentity(g.io, g.allocator, g.model);
    var head: [10]u8 = undefined;
    std.mem.writeInt(u16, head[0..2], @intCast(name.len), .little);
    std.mem.writeInt(u64, head[2..10], id, .little);
    try link.bell.writeAll(&head);
    try link.bell.writeAll(name);
    var fds = [_]std.posix.pollfd{.{ .fd = link.bell.fd, .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&fds, 30_000) == 0) return error.DistNoAnswer;
    var reply: [5]u8 = undefined;
    try link.bell.readAll(&reply);
    cluster().worker_copy.store(reply[0], .release);
    worker_free_mib.store(std.mem.readInt(u32, reply[1..5], .little), .release);
    if (reply[0] != 0) log.err("[dist] the worker does not have {s} ({s}): download the same model there, then join again\n", .{ name, workerCopyName() });
}

/// The worker's side of `askForModel`: looks the model up in its folders and answers with
/// that and its free memory.
fn answerModel(io: std.Io, allocator: std.mem.Allocator, link: *const Link, roots: []const []const u8) !bool {
    if (!link.bell.open()) return true;
    var head: [10]u8 = undefined;
    try link.bell.readAll(&head);
    var name_buf: [256]u8 = undefined;
    const len = std.mem.readInt(u16, head[0..2], .little);
    if (len == 0 or len > name_buf.len) return error.DistBadMessage;
    const name = name_buf[0..len];
    try link.bell.readAll(name);
    var reply: [5]u8 = undefined;
    std.mem.writeInt(u32, reply[1..5], @intCast(@min(status.getAvailableMemBytes() >> 20, std.math.maxInt(u32))), .little);
    const found = findModel(io, allocator, roots, name, std.mem.readInt(u64, head[2..10], .little)) catch |err| switch (err) {
        error.DistModelMismatch => {
            log.err("[dist] worker: {s} here is not the host's copy (quantization or revision)\n", .{name});
            reply[0] = @intCast(MISMATCH);
            try link.bell.writeAll(&reply);
            return false;
        },
        else => return err,
    };
    if (found) |dir| {
        allocator.free(dir);
        reply[0] = 0;
        try link.bell.writeAll(&reply);
        return true;
    }
    log.err("[dist] worker: no model named {s} in {s}\n", .{ name, roots[0] });
    reply[0] = @intCast(NOT_FOUND);
    try link.bell.writeAll(&reply);
    return false;
}

/// The folder named `name` in `roots` (`<root>/<name>` or `<root>/<org>/<name>`) whose
/// `modelIdentity` is `id`; `error.DistModelMismatch` when only other copies of that name exist.
fn findModel(io: std.Io, allocator: std.mem.Allocator, roots: []const []const u8, name: []const u8, id: u64) !?[]u8 {
    var seen = false;
    for (roots) |root| {
        if (try sameModel(io, allocator, &.{ root, name }, id, &seen)) |p| return p;
        var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |e| {
            if (e.kind != .directory) continue;
            if (try sameModel(io, allocator, &.{ root, e.name, name }, id, &seen)) |p| return p;
        }
    }
    return if (seen) error.DistModelMismatch else null;
}

fn sameModel(io: std.Io, allocator: std.mem.Allocator, parts: []const []const u8, id: u64, seen: *bool) !?[]u8 {
    const p = try std.fs.path.join(allocator, parts);
    errdefer allocator.free(p);
    const here = modelIdentity(io, allocator, p) catch {
        allocator.free(p);
        return null;
    };
    seen.* = true;
    if (here == id) return p;
    allocator.free(p);
    return null;
}

/// `mlx-serve worker --coordinator IP:PORT --device rdma_enX [--model-dir DIR]...`: joins
/// rank 0's group, loads the part of the model rank 0 names (found by folder name in its own
/// model folders, `~/.mlx-serve/models` by default), then runs forwards until rank 0 stops it.
pub fn cmdWorker(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !void {
    // JACCL (RDMA over Thunderbolt) exists only on macOS.
    if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable;
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
        home_root = try @import("../../../host/cli/cli.zig").modelsRootPath(allocator, std.mem.span(env.get(.HOME) orelse "/tmp"));
        try roots.append(allocator, home_root.?);
    }
    try rest.appendSlice(allocator, &.{ "--rank", "1" });
    const cfg = try parseArgs(rest.items);

    // A failed JACCL init leaks its socket, so a retry is a fresh process: the app restarts
    // the worker a few seconds later (headless: run it under launchd or in a loop).
    var link = Link.open(io, cfg) catch |err| {
        if (err == error.RdmaInitFailed) log.info("[dist] worker: the host at {s} is not listening yet; retrying\n", .{cfg.coordinator});
        return err;
    };
    defer link.deinit();
    var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const host = std.posix.gethostname(&host_buf) catch "mac";
    const short = host[0..@min(host.len, 64)];
    if (!try answerModel(io, allocator, &link, roots.items)) {
        // The host refuses to use this worker; sleep until it leaves (unpair, restart).
        while (true) try link.bell.wait();
    }
    log.info("[dist] worker: joined, waiting for the host to load a model\n", .{});
    // The host rings when it reads the hello, which it does when it first loads the model.
    try link.bell.wait();
    try sendMsg(&link, Hello.here(short.len).msg(), 0);
    if (short.len > 0) try sendBytes(&link, short, 0);
    while (true) {
        const pmsg = try recvMsg(&link, 0);
        if (pmsg[0] == OP_STOP or !try servePlan(io, allocator, &link, roots.items, pmsg)) break;
        log.info("[dist] worker: released its part, waiting for the host to load a model\n", .{});
    }
}

/// One plan: finds and loads this Mac's part, then serves until rank 0 stops it (false) or
/// releases it (true: the host unloaded the model and may plan again).
fn servePlan(io: std.Io, allocator: std.mem.Allocator, link: *const Link, roots: []const []const u8, pmsg: Msg) !bool {
    const pl = try Plan.parse(pmsg);
    var name_buf: [256]u8 = undefined;
    const name_len: usize = @intCast(pmsg[5]);
    if (name_len == 0 or name_len > name_buf.len) return error.DistBadMessage;
    try recvBytes(link, name_buf[0..name_len], 0);
    const name = name_buf[0..name_len];
    const idm = try recvMsg(link, 0);
    if (idm[0] != MAGIC) return error.DistProtocolMismatch;
    const id = @as(u64, @as(u32, @bitCast(idm[1]))) | @as(u64, @as(u32, @bitCast(idm[2]))) << 32;
    switch (pl.mode) {
        .pipeline => log.info("[dist] worker: {s} layers {d}..{d} of {d} (pipeline)\n", .{ name, pl.span.start, pl.span.end, pl.span.total }),
        .tensor => log.info("[dist] worker: {s}, half of every expert (tensor)\n", .{name}),
        .rows => log.info("[dist] worker: {s}, the whole DiT (video steps)\n", .{name}),
    }

    const fail = struct {
        fn send(l: *const Link, code: i32) void {
            sendMsg(l, .{ MAGIC, code, 0, 0, 0, 0, 0, 0 }, 0) catch {};
        }
    };
    const found = findModel(io, allocator, roots, name, id) catch |err| {
        if (err == error.DistModelMismatch) log.err("[dist] worker: {s} here is not the host's copy (quantization or revision)\n", .{name});
        fail.send(link, if (err == error.DistModelMismatch) MISMATCH else LOAD_FAILED);
        return err;
    };
    const dir = found orelse {
        log.err("[dist] worker: no model named {s} in {s}\n", .{ name, roots[0] });
        fail.send(link, NOT_FOUND);
        return error.DistModelNotFound;
    };
    defer allocator.free(dir);
    if (pl.mode == .rows) {
        const wl = wireOf(link);
        const released = h3_split.serveWorker(allocator, io, &wl, dir, pl.hidden, @intCast(pl.span.total)) catch |err| {
            log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
            fail.send(link, if (err == error.DistModelMismatch) MISMATCH else LOAD_FAILED);
            return err;
        };
        if (!released) log.info("[dist] worker: stopped by rank 0\n", .{});
        return released;
    }
    var config = try model_mod.parseConfig(io, allocator, dir);
    defer config.deinit(allocator);
    if (!pl.matches(&config)) {
        log.err("[dist] worker: {s} here has a different geometry than the host's\n", .{dir});
        fail.send(link, MISMATCH);
        return error.DistModelMismatch;
    }
    if (config.isDsv41()) return serveDsv41(io, allocator, link, dir, &config, pl);
    var xfm = load(io, allocator, dir, &config, pl.mode, pl.span) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        fail.send(link, LOAD_FAILED);
        return err;
    };
    defer xfm.deinit(allocator);
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try sendMsg(link, .{ MAGIC, 0, mib(active), 0, 0, 0, 0, 0 }, 0);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});

    const t = &xfm.t;
    var ctx: ForwardCtx = .{ .cache = &xfm.cache, .moe_seq_offset = &xfm.off, .ssm_entries = xfm.entries, .capture_hidden = null, .vision_embeddings = null };
    var forwards: u64 = 0;
    while (true) {
        const m = try recvMsg(link, 0);
        switch (m[0]) {
            OP_STOP => break,
            OP_RELEASE => return true,
            OP_FORWARD => {
                const offset: usize = @as(usize, @intCast(m[3])) | (@as(usize, @intCast(m[4])) << 31);
                if (offset == 0) {
                    try xfm.reset();
                } else if (offset != xfm.off) {
                    log.err("[dist] forward at {d} but this worker holds {d} tokens\n", .{ offset, xfm.off });
                    return error.DistPeerDiverged;
                }
                const shape = [_]c_int{ m[1], m[2], m[5] };
                const h = try queuedRecv(link, &shape, @fromBackingInt(@intCast(m[6])), 0);
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
                if (m[1] * m[2] > 1 or forwards % 256 == 0) log.info("[dist] worker: forward {d}: {d}x{d} rows, now at {d} tokens\n", .{ forwards, m[1], m[2], xfm.off });
            },
            OP_MOE => {
                const x = try queuedRecv(link, &.{ m[1], m[2], m[3] }, @fromBackingInt(@intCast(m[5])), 0);
                defer _ = mlx.mlx_array_free(x);
                const logits = try queuedRecv(link, &.{ m[1], m[2], m[4] }, @fromBackingInt(@intCast(m[7])), 0);
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
    return false;
}

/// The worker's part of the pack: its weights, the trunk module over them and the one request's caches.
const Loaded = struct {
    w: model_mod.Weights,
    t: Qwen4Trunk,
    cache: KVCache,
    entries: []SSMCacheEntry,
    off: usize = 0,

    /// A new request: no keys, no recurrent state.
    fn reset(self: *Loaded) !void {
        try self.cache.reinit(self.t.config.num_hidden_layers, self.cache.config);
        for (self.entries) |*e| {
            freeEntry(e);
            e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        }
        self.off = 0;
    }

    fn deinit(self: *Loaded, a: std.mem.Allocator) void {
        for (self.entries) |*e| freeEntry(e);
        a.free(self.entries);
        self.cache.deinit();
        self.t.deinit();
        self.w.deinit();
    }

    fn freeEntry(e: *SSMCacheEntry) void {
        ssmFreeSpecCapture(e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    }
};

fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, config: *model_mod.ModelConfig, mode: Mode, span: LayerSpan) !Loaded {
    if (!config.isQwen4()) return error.DistArchUnsupported;
    switch (mode) {
        .pipeline => try config.applyLayerSpan(span),
        .tensor => try config.applyShard(.{ .rank = 1, .size = 2 }),
        .rows => return error.DistArchUnsupported,
    }
    var w = try model_mod.loadModelWeights(io, allocator, dir, config, false);
    errdefer w.deinit();
    var t = try Qwen4Trunk.init(io, allocator, config.*, &w);
    errdefer t.deinit();
    _ = mlx.applyWiredPolicy();
    t.compileMoeRouting();
    if (config.linear_num_key_heads > 0) t.compileGdnGate();
    var cache = try KVCache.init(allocator, config.num_hidden_layers);
    errdefer cache.deinit();
    const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return .{ .w = w, .t = t, .cache = cache, .entries = entries };
}

fn evalOne(a: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

/// `mlx-serve dist-probe --rank R --coordinator IP:PORT --device rdma_enX`:
/// joins the group and measures round trips at the sizes inference moves
/// (one decode row of a hyper-connection stream, a prefill chunk) and all_sum.
pub fn cmdProbe(io: std.Io, args: []const []const u8) !void {
    // `--soak N`: N header + one-row-stream round trips, the shape of a pipeline decode hop.
    var soak: u64 = 0;
    var rest: [16][]const u8 = undefined;
    var n_rest: usize = 0;
    var ai: usize = 0;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--soak") and ai + 1 < args.len) {
            soak = std.fmt.parseInt(u64, args[ai + 1], 10) catch return error.InvalidArgs;
            ai += 1;
        } else if (n_rest < rest.len) {
            rest[n_rest] = args[ai];
            n_rest += 1;
        }
    }
    const cfg = try parseArgs(rest[0..n_rest]);
    var link = try Link.open(io, cfg);
    defer link.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const other: u32 = if (link.rank == 0) 1 else 0;
    if (soak > 0) return soakLink(io, &link, other, soak, s);

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

fn soakLink(io: std.Io, link: *const Link, other: u32, n: u64, s: mlx.mlx_stream) !void {
    // Both ranks draw the same sizes: 1..2400 rows of a 20 KB stream, the range real hops span.
    var rng = std.Random.DefaultPrng.init(0x5eed);
    var sw = io_util.Stopwatch.init(io);
    var bytes: u64 = 0;
    for (0..n) |k| {
        const rows: c_int = rng.random().intRangeAtMost(c_int, 1, 2400);
        const shape = [_]c_int{ 1, rows, 10240 };
        if (link.rank == 0) {
            var x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&x, &shape, 3, .bfloat16, s));
            defer _ = mlx.mlx_array_free(x);
            try evalOne(x);
            try sendMsg(link, .{ OP_FORWARD, rows, @intCast(k & 0x7fffffff), 0, 0, 0, 0, 0 }, other);
            const sent = try link.send(x, other);
            defer _ = mlx.mlx_array_free(sent);
            try evalOne(sent);
            const back = try link.recv(&shape, .bfloat16, other);
            defer _ = mlx.mlx_array_free(back);
            try evalOne(back);
        } else {
            const m = try recvMsg(link, other);
            if (m[1] != rows) return error.DistBadMessage;
            const got = try link.recv(&shape, .bfloat16, other);
            defer _ = mlx.mlx_array_free(got);
            try evalOne(got);
            const sent = try link.send(got, other);
            defer _ = mlx.mlx_array_free(sent);
            try evalOne(sent);
        }
        bytes += @as(u64, @intCast(rows)) * 20480;
        if ((k + 1) % 1000 == 0) {
            log.info("[dist-probe] soak: {d} round trips, {d} MB each way in {d} ms\n", .{ k + 1, bytes >> 20, sw.read() / std.time.ns_per_ms });
            sw = io_util.Stopwatch.init(io);
            bytes = 0;
        }
    }
    log.info("[dist-probe] soak: {d} round trips clean\n", .{n});
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

test "clusterJson: valid JSON naming both ranks' layers and the worker" {
    const a = std.testing.allocator;
    var p: Peer = .{ .link = undefined, .mode = .pipeline, .model = "/m/org/Flash-FP4", .split = null, .worker = .{ .avail_mib = 100, .total_mib = 128 }, .span = .{ .start = 0, .end = 16, .total = 48 }, .planned = true, .ready = true, .exchanges = 3 };
    const name = "Mac \"two\"";
    @memcpy(p.worker_name_buf[0..name.len], name);
    p.worker_name_len = name.len;
    const j = try clusterJson(a, &p);
    defer a.free(j);
    const doc = try std.fmt.allocPrint(a, "{{{s}}}", .{j[1..]});
    defer a.free(doc);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, doc, .{});
    defer parsed.deinit();
    const c = parsed.value.object.get("cluster").?.object;
    try std.testing.expectEqualStrings("pipeline", c.get("mode").?.string);
    try std.testing.expectEqualStrings("Flash-FP4", c.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 16), c.get("here").?.array.items[1].integer);
    try std.testing.expectEqualStrings(name, c.get("worker_name").?.string);
}

test "findModel pairs the host's copy of a name, not another quantization of it" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "org-a/Flash", "org-b/Flash" }, [_][]const u8{ "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
        const w = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir});
        defer allocator.free(w);
        try tmp.dir.writeFile(io, .{ .sub_path = w, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const root = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(root);
    const b = try std.fmt.allocPrint(allocator, "{s}/org-b/Flash", .{root});
    defer allocator.free(b);

    const found = (try findModel(io, allocator, &.{root}, "Flash", try modelIdentity(io, allocator, b))).?;
    defer allocator.free(found);
    try std.testing.expectEqualStrings(b, found);
    try std.testing.expectError(error.DistModelMismatch, findModel(io, allocator, &.{root}, "Flash", 42));
    try std.testing.expect((try findModel(io, allocator, &.{root}, "Other", 42)) == null);
}

test "pairFields names the model a Thunderbolt pair leads, as valid JSON" {
    const a = std.testing.allocator;
    const f = try pairFieldsFor(a, "MiniMax \"H3\"", false);
    defer a.free(f);
    const j = try std.fmt.allocPrint(a, "{{\"os\":\"macos\"{s}}}", .{f});
    defer a.free(j);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, j, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("MiniMax \"H3\"", parsed.value.object.get("leads").?.string);
    try std.testing.expect(!parsed.value.object.get("joined").?.bool);
}

test "a plan made for one copy refuses another folder of the same name" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "a/Flash", "b/Flash" }, [_][]const u8{ "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const a = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}/a/Flash", .{ cwd, tmp.sub_path });
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}/b/Flash", .{ cwd, tmp.sub_path });
    defer allocator.free(b);

    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = a, .split = null };
    defer cluster().group = null;
    var p: Peer = undefined;
    p.plan_id = try modelIdentity(io, allocator, a);
    try p.samePlannedCopy(a);
    try std.testing.expectError(error.DistModelMismatch, p.samePlannedCopy(b));
}

test "the pair holds its own folder, not another one with the same name" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "a/Flash");
    try tmp.dir.createDirPath(io, "b/Flash");
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const a = try std.fmt.allocPrint(allocator, "{s}/a/Flash", .{root});
    defer allocator.free(a);
    const a_spelled = try std.fmt.allocPrint(allocator, "{s}/b/../a/Flash/", .{root});
    defer allocator.free(a_spelled);
    const b = try std.fmt.allocPrint(allocator, "{s}/b/Flash", .{root});
    defer allocator.free(b);

    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = a, .split = null };
    defer cluster().group = null;
    cluster().joined.store(true, .release);
    defer cluster().joined.store(false, .release);
    try std.testing.expect(holds(a));
    try std.testing.expect(holds(a_spelled));
    try std.testing.expect(!holds(b));
    // The other folder loads, and is billed, whole.
    var c: model_mod.ModelConfig = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .num_hidden_layers = 8 };
    worker_free_mib.store(40, .release);
    defer worker_free_mib.store(0, .release);
    try std.testing.expectEqual(@as(u64, 800), hostShare(b, &c, 800));
}

test "parseCoordinator reads the host's address and port" {
    const c = try parseCoordinator("10.254.77.1:32323");
    try std.testing.expectEqual([4]u8{ 10, 254, 77, 1 }, c.ip4);
    try std.testing.expectEqual(@as(u16, 32323), c.port);
    try std.testing.expectError(error.DistBadCoordinator, parseCoordinator("10.254.77:32323"));
    try std.testing.expectError(error.DistBadCoordinator, parseCoordinator("10.254.77.1"));
}

test "the bell wakes rank 1 once per ring and reports a host that left" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 39411 };
    const listener = Bell.listen(coord);
    try std.testing.expect(listener >= 0);
    defer _ = std.c.close(listener);
    const worker = Bell.connect(coord);
    try std.testing.expect(worker.open());
    const host = Bell.accept(listener);
    try std.testing.expect(host.open());
    try host.ring();
    try host.ring();
    try worker.wait();
    try worker.wait();
    _ = std.c.close(host.fd);
    try std.testing.expectError(error.DistPeerGone, worker.wait());
    _ = std.c.close(worker.fd);
}

test "the worker answers whether it has the host's copy of the model when it joins" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "host/Flash", "worker/Flash", "other/Flash" }, [_][]const u8{ "4bit", "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const base = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(base);
    const host_model = try std.fmt.allocPrint(allocator, "{s}/host/Flash", .{base});
    defer allocator.free(host_model);
    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = host_model, .split = null };
    defer cluster().group = null;
    defer cluster().worker_copy.store(-1, .release);

    const Case = struct { root: []const u8, want: i32 };
    for ([_]Case{ .{ .root = "worker", .want = 0 }, .{ .root = "other", .want = MISMATCH }, .{ .root = "nowhere", .want = NOT_FOUND } }) |c| {
        const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 39421 };
        const listener = Bell.listen(coord);
        try std.testing.expect(listener >= 0);
        defer _ = std.c.close(listener);
        var worker: Link = .{ .group = undefined, .rank = 1, .size = 2, .bell = Bell.connect(coord) };
        defer _ = std.c.close(worker.bell.fd);
        var host: Link = .{ .group = undefined, .rank = 0, .size = 2, .bell = Bell.accept(listener) };
        defer _ = std.c.close(host.bell.fd);
        const asker = try std.Thread.spawn(.{}, askForModel, .{&host});
        const root = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, c.root });
        defer allocator.free(root);
        const has = try answerModel(io, allocator, &worker, &.{root});
        asker.join();
        try std.testing.expectEqual(c.want == 0, has);
        try std.testing.expectEqual(c.want, cluster().worker_copy.load(.acquire));
    }
}

test "a paired model is never picked to unload while the worker is joined" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    const reg_mod = @import("../../../host/registry/registry.zig");
    var reg = try reg_mod.ModelRegistry.init(std.testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const flash = try reg.registerStub("Flash", "/m/Flash", 100);
    const other = try reg.registerStub("other", "other", 100);
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(flash, 100);
    reg.markReadyLocked(other, 100);
    reg.mutex.unlock(reg.io);
    _ = try reg.ensureLoaded("other");
    reg.release(other); // `other` is now the newest; Flash the oldest

    cluster().group = .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .pipeline, .model = "/m/Flash", .split = null };
    defer cluster().group = null;
    cluster().joined.store(true, .release);
    defer cluster().joined.store(false, .release);

    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    try std.testing.expectEqualStrings("other", reg.pickLruEvictable("", false).?.id);
    try std.testing.expect(!reg_mod.ModelRegistry.idleEvictable(flash, std.math.maxInt(i64), 0));
    cluster().joined.store(false, .release);
    try std.testing.expectEqualStrings("Flash", reg.pickLruEvictable("", false).?.id);
}

test "the leader bills only the layers it keeps of a model its pair splits" {
    var c: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8 };
    c.dsv4_compress_ratios[0..8].* = .{ 0, 0, 2, 2, 2, 1, 1, 1 };
    c.dsv41_kv_sources = (1 << 2) | (1 << 5);
    c.dsv41_index_sources = (1 << 2) | (1 << 5) | (1 << 7);
    c.dsv41_candidate_source = 5;
    c.dsv41_engram_layers[0..2].* = .{ 1, 2 };
    c.dsv41_n_engram_layers = 2;
    const g: Group = .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .pipeline, .model = "/m/V41", .split = null };
    // The worker's 40% holds layers 5..8: this Mac keeps 5 of 8.
    try std.testing.expectEqual(@as(u64, 500), shareFor(g, &c, 800, 60, 40));
    // A worker that has not said how much it has: the whole model.
    try std.testing.expectEqual(@as(u64, 800), shareFor(g, &c, 800, 60, 0));
}

test "after the stop message rank 0 sends the worker nothing more" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    // A read-only bell fails to ring, so no message reaches the (absent) RDMA link.
    const fd = std.c.open("/dev/null", .{ .ACCMODE = .RDONLY });
    defer _ = std.c.close(fd);
    cluster().peer_slot = .{ .link = .{ .group = undefined, .rank = 0, .size = 2, .bell = .{ .fd = fd } }, .mode = .pipeline, .model = "/m/H3", .split = null, .worker = undefined, .ready = true };
    defer cluster().peer_slot = null;
    shutdown();
    try std.testing.expect(!peer().?.ready); // the model's unload at teardown must not message a stopped worker
}
