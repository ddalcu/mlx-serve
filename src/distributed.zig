//! Multi-Mac inference over MLX's distributed group (JACCL = RDMA over
//! Thunderbolt 5, macOS 26.2+). Rank 0 is the server; every other rank is a
//! worker that executes what rank 0 tells it, in lockstep.
const std = @import("std");
const mlx = @import("mlx.zig");
const crt = @import("crt.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

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

    /// Lazy send of `x` to `dst`; the returned array must be evaluated (or depended on).
    pub fn send(self: *const Link, x: mlx.mlx_array, dst: u32, s: mlx.mlx_stream) !mlx.mlx_array {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_send(&r, x, @intCast(dst), self.group, s));
        return r;
    }

    pub fn recv(self: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32, s: mlx.mlx_stream) !mlx.mlx_array {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_recv(&r, shape.ptr, shape.len, dtype, @intCast(src), self.group, s));
        return r;
    }

    pub fn allSum(self: *const Link, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_all_sum(&r, x, self.group, s));
        return r;
    }
};

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
    const peer: u32 = if (link.rank == 0) 1 else 0;

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
                const sent = try link.send(x, peer, s);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                const back = try link.recv(&shape, .float32, peer, s);
                defer _ = mlx.mlx_array_free(back);
                try evalOne(back);
            } else {
                const got = try link.recv(&shape, .float32, peer, s);
                defer _ = mlx.mlx_array_free(got);
                try evalOne(got);
                const sent = try link.send(got, peer, s);
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
