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
        const head_only = std.mem.startsWith(u8, key, "mtp.") or std.mem.indexOf(u8, key, ".mtp.") != null or
            std.mem.indexOf(u8, key, "visual") != null or std.mem.indexOf(u8, key, "vision") != null or
            std.mem.indexOf(u8, key, "audio") != null;
        if (head_only) return if (self.first()) key else null;
        const at = trunkLayerAt(key) orelse return key;
        const n = std.fmt.parseInt(u32, key[at.lo..at.hi], 10) catch return key;
        if (n < self.start or n >= self.end) return null;
        if (self.start == 0) return key;
        return std.fmt.bufPrint(buf, "{s}{d}{s}", .{ key[0..at.lo], n - self.start, key[at.hi..] }) catch null;
    }
};

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
    pub fn send(self: *const Link, x: mlx.mlx_array, dst: u32, _: mlx.mlx_stream) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_send(&r, x, @intCast(dst), self.group, cs));
        return r;
    }

    pub fn recv(self: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32, _: mlx.mlx_stream) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_recv(&r, shape.ptr, shape.len, dtype, @intCast(src), self.group, cs));
        return r;
    }

    pub fn allSum(self: *const Link, x: mlx.mlx_array, _: mlx.mlx_stream) !mlx.mlx_array {
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

fn sendMsg(link: *const Link, m: Msg, dst: u32, s: mlx.mlx_stream) !void {
    const shape = [_]c_int{8};
    const a = mlx.mlx_array_new_data(&m, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst, s);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvMsg(link: *const Link, src: u32, s: mlx.mlx_stream) !Msg {
    const r = try link.recv(&.{8}, .int32, src, s);
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

/// What a worker reports before it loads: its memory and the model it found on its disk.
const Hello = struct {
    avail_mib: i32,
    total_mib: i32,
    layers: i32,
    hidden: i32,
    vocab: i32,
    experts: i32,

    fn of(cfg: *const model_mod.ModelConfig) Hello {
        return .{ .avail_mib = mib(status.getAvailableMemBytes()), .total_mib = mib(status.getTotalMemBytes()), .layers = @intCast(cfg.num_hidden_layers), .hidden = @intCast(cfg.hidden_size), .vocab = @intCast(cfg.vocab_size), .experts = @intCast(cfg.num_experts) };
    }
    fn msg(h: Hello) Msg {
        return .{ MAGIC, VERSION, h.avail_mib, h.total_mib, h.layers, h.hidden, h.vocab, h.experts };
    }
    fn parse(m: Msg) !Hello {
        if (m[0] != MAGIC or m[1] != VERSION) return error.DistProtocolMismatch;
        return .{ .avail_mib = m[2], .total_mib = m[3], .layers = m[4], .hidden = m[5], .vocab = m[6], .experts = m[7] };
    }
    fn sameModel(a: Hello, b: Hello) bool {
        return a.layers == b.layers and a.hidden == b.hidden and a.vocab == b.vocab and a.experts == b.experts;
    }
};

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
    span: LayerSpan,
    model_dir: []const u8,
    ready: bool = false,
    /// Tokens the worker's caches hold. Every forward must continue them or start over at 0.
    peer_pos: usize = 0,

    /// Hands the stream to the worker and returns its output stream. Lazy: the sends are
    /// queued (async) ahead of the receive so the comm stream runs them in that order.
    pub fn hop(self: *Peer, h: mlx.mlx_array, offset: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        if (!self.ready) try self.awaitReady(s);
        if (offset != 0 and offset != self.peer_pos) {
            log.err("[dist] forward at {d} but the worker holds {d} tokens: its caches cannot follow this reuse\n", .{ offset, self.peer_pos });
            return error.DistPeerDiverged;
        }
        const sh = mlx.getShape(h);
        const hdr_host: Msg = .{ OP_FORWARD, sh[0], sh[1], @intCast(offset & 0x7fffffff), @intCast(offset >> 31), sh[2], @intFromEnum(mlx.mlx_array_dtype(h)), 0 };
        const shape = [_]c_int{8};
        const hdr = mlx.mlx_array_new_data(&hdr_host, &shape, 1, .int32);
        defer _ = mlx.mlx_array_free(hdr);
        const hdr_sent = try self.link.send(hdr, 1, s);
        defer _ = mlx.mlx_array_free(hdr_sent);
        const ordered = try dependsOn(h, hdr_sent);
        defer _ = mlx.mlx_array_free(ordered);
        const sent = try self.link.send(ordered, 1, s);
        defer _ = mlx.mlx_array_free(sent);
        {
            const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{sent}, 1);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
        }
        self.peer_pos = offset + @as(usize, @intCast(sh[1]));
        return self.link.recv(sh, mlx.mlx_array_dtype(h), 1, s);
    }

    fn awaitReady(self: *Peer, s: mlx.mlx_stream) !void {
        const m = try recvMsg(&self.link, 1, s);
        if (m[0] != MAGIC or m[1] != 0) {
            log.err("[dist] the worker failed to load its layers (code {d})\n", .{m[1]});
            return error.DistWorkerLoadFailed;
        }
        log.info("[dist] worker ready: layers {d}..{d} resident, {d} MiB\n", .{ self.span.end, self.span.total, m[2] });
        self.ready = true;
    }
};

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

/// Rank 0's peer, once `connect` has run.
pub fn peer() ?*Peer {
    return if (peer_slot) |*p| p else null;
}

/// The trunk span this process serves for `model_dir`, if it is the distributed model.
pub fn spanFor(model_dir: []const u8) ?LayerSpan {
    const p = peer() orelse return null;
    if (!std.mem.eql(u8, std.fs.path.basename(p.model_dir), std.fs.path.basename(model_dir))) return null;
    return p.span;
}

/// Rank 0 at startup: joins the group, checks the worker found the same model,
/// splits the layers by free memory (or at `split`), and tells the worker its span.
pub fn connect(io: std.Io, allocator: std.mem.Allocator, cfg: Config, mode: Mode, model_dir: []const u8, split: ?u32) !void {
    if (mode != .pipeline) return error.DistModeUnsupported;
    var link = try Link.open(io, cfg);
    errdefer link.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var mc = try model_mod.parseConfig(io, allocator, model_dir);
    defer mc.deinit(allocator);
    const mine = Hello.of(&mc);
    const theirs = try Hello.parse(try recvMsg(&link, 1, s));
    if (!mine.sameModel(theirs)) {
        log.err("[dist] the worker loaded a different model ({d} layers, hidden {d}, vocab {d}) than {s}\n", .{ theirs.layers, theirs.hidden, theirs.vocab, model_dir });
        return error.DistModelMismatch;
    }
    const n = mc.num_hidden_layers;
    const k = split orelse chooseSplit(n, mc.full_attention_interval, @as(u64, @intCast(mine.avail_mib)), @as(u64, @intCast(theirs.avail_mib)));
    if (k == 0 or k >= n or (mc.full_attention_interval != 0 and k % mc.full_attention_interval != 0)) return error.DistSplitOffInterval;
    try sendMsg(&link, .{ MAGIC, @intFromEnum(mode), @intCast(k), @intCast(n), @intCast(n), 0, 0, 0 }, 1, s);
    log.info("[dist] pipeline: layers 0..{d} here ({d} MiB free), {d}..{d} on the worker ({d} MiB free of {d})\n", .{ k, mine.avail_mib, k, n, theirs.avail_mib, theirs.total_mib });
    peer_slot = .{ .link = link, .mode = mode, .span = .{ .start = 0, .end = k, .total = n }, .model_dir = try allocator.dupe(u8, model_dir) };
}

/// `mlx-serve worker --model DIR --coordinator IP:PORT --device rdma_enX`: joins rank 0's
/// group, loads the span it is given, then runs forwards until rank 0 stops it.
pub fn cmdWorker(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !void {
    var model_dir: ?[]const u8 = null;
    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(allocator);
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            model_dir = args[i + 1];
            i += 1;
        } else try rest.append(allocator, args[i]);
    }
    try rest.appendSlice(allocator, &.{ "--rank", "1" });
    const cfg = try parseArgs(rest.items);
    const dir = model_dir orelse {
        log.err("usage: mlx-serve worker --model DIR --coordinator IP:PORT --device rdma_enX\n", .{});
        return error.InvalidArgs;
    };
    var config = try model_mod.parseConfig(io, allocator, dir);
    defer config.deinit(allocator);

    var link = try Link.open(io, cfg);
    defer link.deinit();
    const cs = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cs);
    try sendMsg(&link, Hello.of(&config).msg(), 0, cs);
    const plan = try recvMsg(&link, 0, cs);
    if (plan[0] != MAGIC) return error.DistProtocolMismatch;
    const span: LayerSpan = .{ .start = @intCast(plan[2]), .end = @intCast(plan[3]), .total = @intCast(plan[4]) };
    log.info("[dist] worker: layers {d}..{d} of {d} ({s})\n", .{ span.start, span.end, span.total, @tagName(@as(Mode, @enumFromInt(plan[1]))) });

    var xfm = load(io, allocator, dir, &config, span) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        sendMsg(&link, .{ MAGIC, 1, 0, 0, 0, 0, 0, 0 }, 0, cs) catch {};
        return err;
    };
    defer xfm.t.deinit();
    defer xfm.w.deinit();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try sendMsg(&link, .{ MAGIC, 0, mib(active), 0, 0, 0, 0, 0 }, 0, cs);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});

    const t = &xfm.t;
    var ctx = t.defaultCtx();
    var forwards: u64 = 0;
    while (true) {
        const m = try recvMsg(&link, 0, cs);
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
                const h = try link.recv(&shape, @enumFromInt(m[6]), 0, t.s);
                defer _ = mlx.mlx_array_free(h);
                const out = try t.forwardQwen4Stream(&ctx, h);
                defer _ = mlx.mlx_array_free(out);
                const sent = try link.send(out, 0, t.s);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                forwards += 1;
                if (forwards % 256 == 0) log.info("[dist] worker: {d} forwards, at {d} tokens\n", .{ forwards, t.moe_seq_offset });
            },
            else => return error.DistBadMessage,
        }
    }
    log.info("[dist] worker: stopped by rank 0 after {d} forwards\n", .{forwards});
}

const Loaded = struct { w: model_mod.Weights, t: transformer_mod.Transformer };

fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, config: *model_mod.ModelConfig, span: LayerSpan) !Loaded {
    if (!config.isQwen4()) return error.DistArchUnsupported;
    try config.applyLayerSpan(span);
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
                const sent = try link.send(x, other, s);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                const back = try link.recv(&shape, .float32, other, s);
                defer _ = mlx.mlx_array_free(back);
                try evalOne(back);
            } else {
                const got = try link.recv(&shape, .float32, other, s);
                defer _ = mlx.mlx_array_free(got);
                try evalOne(got);
                const sent = try link.send(got, other, s);
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
            const r = try link.allSum(x, s);
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
