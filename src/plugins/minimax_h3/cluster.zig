//! MiniMax-H3 on two Macs over Thunderbolt RDMA: both hold the whole DiT and each
//! sampling step's rows are split between them. A block runs on a Mac's own rows;
//! its attention reads every row's keys and values, swapped once per block.
//! Worth it only between Macs of similar speed: `chooseCut` decides per job.
const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const wire = @import("../../sdk/util/wire.zig");
const h3 = @import("h3.zig");
const config = @import("config.zig");

const S = mlx.mlx_stream;

/// Rows the speed calibration runs: enough that attention is part of the cost.
pub const CALIB_ROWS: u32 = 2048;
/// What one block's swap costs beyond its bytes: two evals and the collective's handshake.
const BLOCK_SYNC_US: f64 = 500;
/// A split must make a step at least this much faster, or the job runs on this Mac alone.
const MIN_GAIN: f64 = 1.15;

/// One job's split: rows `[0, cut)` on rank 0, `[cut, rows)` on rank 1.
pub const RowSplit = struct {
    link: *const wire.Link,
    rank: u32,
    rows: u32,
    cut: u32,

    fn count(self: *const RowSplit, r: u32) u32 {
        return if (r == 0) self.cut else self.rows - self.cut;
    }

    /// Every row's keys and values from this Mac's `[1, H, n, hd]` pair, rank 0's rows first.
    /// all_gather wants equal shapes, so the shorter side pads (and the padding is cut out).
    pub fn gatherKv(self: *const RowSplit, k: mlx.mlx_array, v: mlx.mlx_array, s: S) ![2]mlx.mlx_array {
        const n0: c_int = @intCast(self.count(0));
        const n1: c_int = @intCast(self.count(1));
        const nmax = @max(n0, n1);
        const mine: c_int = @intCast(self.count(self.rank));
        var kv = try cat(&.{ k, v }, 0, s);
        defer _ = mlx.mlx_array_free(kv);
        if (mine < nmax) {
            const sh = mlx.getShape(kv);
            var z = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(z);
            try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ sh[0], sh[1], nmax - mine, sh[3] }, 4, mlx.mlx_array_dtype(kv), s));
            const padded = try cat(&.{ kv, z }, 2, s);
            _ = mlx.mlx_array_free(kv);
            kv = padded;
        }
        // Evaluated before the swap and the swap before anything reads it: a lazy collective
        // inside a GPU command buffer can wait on the very buffer that feeds it.
        try mlx.check(mlx.mlx_array_eval(kv));
        const g = try self.link.allGather(kv);
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_array_eval(g));
        var out: [2]mlx.mlx_array = undefined;
        for (0..2) |i| {
            const a = try slice4(g, @intCast(i), n0, s);
            defer _ = mlx.mlx_array_free(a);
            const b = try slice4(g, @intCast(i + 2), n1, s);
            defer _ = mlx.mlx_array_free(b);
            out[i] = try cat(&.{ a, b }, 2, s);
        }
        return out;
    }

    /// Rank 0, each step: hands the worker its rows and what modulates them, and returns this
    /// Mac's rows with their runs and RoPE rows.
    pub fn shareRows(self: *const RowSplit, h: mlx.mlx_array, t_emb: mlx.mlx_array, runs: []const h3.ModRun, rope: h3.RopeTables, refresh: bool, a: std.mem.Allocator, s: S) !Mine {
        if (t_emb.ctx == null) return error.H3SplitNeedsTimeEmbedding;
        const cut: c_int = @intCast(self.cut);
        const rows: c_int = @intCast(self.rows);
        const theirs = try localRuns(a, runs, self.cut, self.rows);
        defer a.free(theirs);
        const te = mlx.getShape(t_emb);
        const half = mlx.getShape(rope.cos)[3];
        try self.link.sendOp(.{ wire.OP_H3_STEP, rows, cut, @intCast(theirs.len), @intFromBool(refresh), te[0], te[1], half });
        const flat = try a.alloc(i32, theirs.len * 3);
        defer a.free(flat);
        for (theirs, 0..) |r, i| flat[i * 3 ..][0..3].* = .{ @intCast(r.start), @intCast(r.end), @intCast(r.mod_row) };
        const runs_a = mlx.mlx_array_new_data(flat.ptr, &[_]c_int{@intCast(flat.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(runs_a);
        const h1 = try sliceAxis(h, 0, cut, rows, s);
        defer _ = mlx.mlx_array_free(h1);
        const c1 = try sliceAxis(rope.cos, 1, cut, rows, s);
        defer _ = mlx.mlx_array_free(c1);
        const s1 = try sliceAxis(rope.sin, 1, cut, rows, s);
        defer _ = mlx.mlx_array_free(s1);
        for ([_]mlx.mlx_array{ h1, t_emb, runs_a, c1, s1 }) |x| try self.link.sendNow(x, 1);

        var m: Mine = .{ .h = try sliceAxis(h, 0, 0, cut, s), .runs = &.{}, .rope = undefined };
        errdefer _ = mlx.mlx_array_free(m.h);
        m.runs = try localRuns(a, runs, 0, self.cut);
        errdefer a.free(m.runs);
        m.rope = .{ .cos = try sliceAxis(rope.cos, 1, 0, cut, s), .sin = try sliceAxis(rope.sin, 1, 0, cut, s) };
        return m;
    }

    /// Rank 0, each step: this Mac's output rows followed by the worker's.
    pub fn collectRows(self: *const RowSplit, mine: mlx.mlx_array, s: S) !mlx.mlx_array {
        try mlx.check(mlx.mlx_array_eval(mine));
        const sh = mlx.getShape(mine);
        const theirs = try self.link.recvNow(&.{ @intCast(self.count(1)), sh[1] }, mlx.mlx_array_dtype(mine), 1);
        defer _ = mlx.mlx_array_free(theirs);
        return cat(&.{ mine, theirs }, 0, s);
    }
};

/// This Mac's part of a step.
pub const Mine = struct {
    h: mlx.mlx_array,
    runs: []h3.ModRun,
    rope: h3.RopeTables,

    pub fn deinit(self: *Mine, a: std.mem.Allocator) void {
        if (self.h.ctx != null) _ = mlx.mlx_array_free(self.h);
        a.free(self.runs);
        self.rope.deinit();
    }
};

/// The modulation runs over rows `[lo, hi)`, renumbered from `lo`.
pub fn localRuns(a: std.mem.Allocator, runs: []const h3.ModRun, lo: u32, hi: u32) ![]h3.ModRun {
    var out: std.ArrayList(h3.ModRun) = .empty;
    errdefer out.deinit(a);
    for (runs) |r| {
        const st = @max(r.start, lo);
        const en = @min(r.end, hi);
        if (st < en) try out.append(a, .{ .start = st - lo, .end = en - lo, .mod_row = r.mod_row });
    }
    return out.toOwnedSlice(a);
}

/// Rows kept on this Mac, or null when splitting would not make a step `MIN_GAIN` faster.
/// `host_us`/`worker_us`: each Mac's calibration time; a step's time is priced linearly in
/// rows from it (attention grows faster, so a big canvas gains more than this predicts).
/// `link`: bytes per microsecond each way, measured when the worker joined.
pub fn chooseCut(rows: u32, host_us: u64, worker_us: u64, link: f64, kv_row_bytes: u64, blocks: u32) ?u32 {
    if (host_us == 0 or worker_us == 0 or link <= 0 or rows < 2) return null;
    const hu: f64 = @floatFromInt(host_us);
    const wu: f64 = @floatFromInt(worker_us);
    const n: f64 = @floatFromInt(rows);
    const share = wu / (hu + wu); // a faster worker (smaller time) leaves less here
    const cut: u32 = std.math.clamp(@as(u32, @intFromFloat(@round(n * share))), 1, rows - 1);
    const solo = hu * n / @as(f64, @floatFromInt(CALIB_ROWS));
    const widest: f64 = @floatFromInt(@max(cut, rows - cut));
    const swap = @as(f64, @floatFromInt(blocks)) * (widest * @as(f64, @floatFromInt(kv_row_bytes)) / link + BLOCK_SYNC_US);
    const split = @max(solo * @as(f64, @floatFromInt(cut)) / n, wu * (n - @as(f64, @floatFromInt(cut))) / @as(f64, @floatFromInt(CALIB_ROWS))) + swap;
    return if (solo >= split * MIN_GAIN) cut else null;
}

/// Microseconds for every block over `CALIB_ROWS` rows: what `chooseCut` weighs the two Macs by.
pub fn calibrate(model: *const h3.Model, io: std.Io, s: S) !u64 {
    const rows: c_int = @intCast(CALIB_ROWS);
    const hidden: c_int = @intCast(model.cfg.hidden_size);
    const half: c_int = @intCast(model.cfg.rotDim() / 2);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0));
    const t_emb = try model.timeEmbedAs(&.{500.0}, s);
    defer _ = mlx.mlx_array_free(t_emb);
    var rope: h3.RopeTables = .{ .cos = mlx.mlx_array_new(), .sin = mlx.mlx_array_new() };
    defer rope.deinit();
    try mlx.check(mlx.mlx_ones(&rope.cos, &[_]c_int{ 1, rows, 1, half }, 4, model.dtype, s));
    try mlx.check(mlx.mlx_zeros(&rope.sin, &[_]c_int{ 1, rows, 1, half }, 4, model.dtype, s));
    const runs = [_]h3.ModRun{.{ .start = 0, .end = CALIB_ROWS, .mod_row = 0 }};
    var best: u64 = std.math.maxInt(u64);
    for (0..3) |i| {
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_random_normal(&h, &[_]c_int{ rows, hidden }, 2, model.dtype, 0, 1, key, s));
        try mlx.check(mlx.mlx_array_eval(h));
        var sw = io_util.Stopwatch.init(io);
        try model.runBlocks(&h, t_emb, .{ .ctx = null }, &runs, rope, null, null, true, s);
        try mlx.check(mlx.mlx_array_eval(h));
        const us = sw.read() / 1000;
        if (i > 0) best = @min(best, us); // the first run compiles kernels
    }
    return best;
}

/// Bytes per microsecond each way through an all_gather of 16 MB per Mac: both Macs run it
/// once, right after the worker reports ready.
pub fn measureLink(link: *const wire.Link, io: std.Io) !f64 {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const n: c_int = 8 << 20;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_ones(&x, &[_]c_int{n}, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(x));
    var best: u64 = std.math.maxInt(u64);
    for (0..4) |i| {
        var sw = io_util.Stopwatch.init(io);
        const g = try link.allGather(x);
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_array_eval(g));
        if (i > 0) best = @min(best, sw.read() / 1000);
    }
    return @as(f64, @floatFromInt(@as(u64, @intCast(n)) * 2)) / @as(f64, @floatFromInt(@max(best, 1)));
}

/// What a job on this Mac reads and keeps of the engine's pair (`distributed.planRows`).
pub const Pair = struct {
    link: *const wire.Link,
    host_calib_us: *u64,
    link_bytes_per_us: *f64,
    exchanges: *u64,
    worker_calib_us: u64,
    /// The worker's free memory less what its model holds.
    worker_free_mib: u64,
};

/// The engine's cluster, set by `distributed.zig` when this server leads a group.
const Hooks = struct {
    /// Whether the group serves the pack at `dir`.
    serves: ?*const fn (dir: []const u8) bool = null,
    /// The worker told to load the group's video model whole, and ready.
    plan_rows: ?*const fn (dir: []const u8, hidden: i32, blocks: i32) anyerror!Pair = null,
    /// What the last video job did with the worker (`/props` cluster "last_job").
    note_job: ?*const fn (split: [2]u32, line: []const u8) void = null,
};
pub const hooks_shared = @import("../../sdk/util/shared.zig").Shared(Hooks, "h3_cluster", .{});
pub inline fn hooks() *Hooks {
    return hooks_shared.get();
}

fn noteJob(split: [2]u32, comptime fmt: []const u8, args: anytype) void {
    var buf: [160]u8 = undefined;
    if (hooks().note_job) |f| f(split, std.fmt.bufPrint(&buf, fmt, args) catch "");
}

/// The job this Mac starts with the worker, or null (with the reason in `/props`) when this
/// job runs here alone.
pub const Job = struct {
    split: RowSplit,

    pub fn end(self: *Job) void {
        self.split.link.sendOp(.{ wire.OP_H3_END, 0, 0, 0, 0, 0, 0, 0 }) catch {};
    }
};

pub fn begin(pack_dir: []const u8, io: std.Io, model: *h3.Model, turbo: bool, user_loras: usize, resident: bool, rows: u32, bcast: bool, s: S) !?Job {
    const h = hooks();
    if (!(if (h.serves) |f| f(pack_dir) else false)) return null;
    const name = std.fs.path.basename(pack_dir);
    if (!resident) return alone("this Mac could not keep {s} loaded between jobs", .{name});
    if (user_loras > 0) return alone("the request's LoRAs live on this Mac only", .{});
    const p = h.plan_rows.?(pack_dir, @intCast(model.cfg.hidden_size), @intCast(model.blocks.len)) catch |err| switch (err) {
        error.DistWorkerNotJoined => return alone("the worker Mac has not joined", .{}),
        error.DistWorkerLoadFailed => return alone("the worker could not load {s}", .{name}),
        else => return err,
    };
    if (p.host_calib_us.* == 0) {
        p.link_bytes_per_us.* = try measureLink(p.link, io);
        p.host_calib_us.* = try calibrate(model, io, s);
        log.info("[h3-split] this Mac: {d} us per {d}-row pass, the worker: {d} us; link {d:.1} GB/s each way\n", .{ p.host_calib_us.*, CALIB_ROWS, p.worker_calib_us, p.link_bytes_per_us.* / 1000 });
    }
    // The worker reads every row's keys, so its step needs about what this Mac's would.
    const bcast_row: u64 = if (bcast) @as(u64, model.blocks.len) * model.cfg.hidden_size * 2 else 0;
    const need_mib = config.activationBytes(rows, bcast_row) >> 20;
    const free_mib = p.worker_free_mib;
    if (need_mib > free_mib) return alone("{d}-row steps need {d} MiB on the worker, which has {d} MiB free", .{ rows, need_mib, free_mib });
    const kv_row: u64 = 2 * @as(u64, model.cfg.num_attention_heads) * model.cfg.attention_head_dim * 2;
    // Diagnostic: diag h3-split-share=<0..1> keeps that share of the rows here, paying or not.
    const forced: ?u32 = if (env.span(.H3_SPLIT_SHARE)) |v| if (std.fmt.parseFloat(f64, v)) |f|
        std.math.clamp(@as(u32, @intFromFloat(@as(f64, @floatFromInt(rows)) * f)), 1, rows - 1)
    else |_| null else null;
    const cut = forced orelse chooseCut(rows, p.host_calib_us.*, p.worker_calib_us, p.link_bytes_per_us.*, kv_row, @intCast(model.blocks.len)) orelse
        return alone("{d} rows: the swap would cost more than the worker saves", .{rows});
    try p.link.sendOp(.{ wire.OP_H3_JOB, @intFromBool(turbo), @intFromBool(bcast), 0, 0, 0, 0, 0 });
    noteJob(.{ cut, rows - cut }, "split {d}/{d} rows", .{ cut, rows - cut });
    log.info("[h3-split] {d} rows: {d} here, {d} on the worker\n", .{ rows, cut, rows - cut });
    p.exchanges.* += 1;
    return .{ .split = .{ .link = p.link, .rank = 0, .rows = rows, .cut = cut } };
}

fn alone(comptime fmt: []const u8, args: anytype) ?Job {
    noteJob(.{ 0, 0 }, "alone: " ++ fmt, args);
    log.info("[h3-split] this job runs here alone: " ++ fmt ++ "\n", args);
    return null;
}

/// The worker's side: loads the whole DiT from `dir`, reports its speed, then runs its rows of
/// every step until rank 0 stops it.
pub fn serveWorker(allocator: std.mem.Allocator, io: std.Io, link: *const wire.Link, dir: []const u8, hidden: i32, blocks: i32) !bool {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const dit = try std.fs.path.join(allocator, &.{ dir, "transformer.safetensors" });
    defer allocator.free(dit);
    var model = try h3.loadDit(allocator, dit, .{}, .bfloat16, s);
    defer model.deinit();
    if (model.cfg.hidden_size != hidden or model.blocks.len != blocks) return error.DistModelMismatch;
    try model.evalWeights();
    var stack: @import("../../sdk/weights/lora.zig").Stack = .{ .allocator = allocator };
    defer stack.deinit();
    var turbo_bound = false;
    const calib = try calibrate(&model, io, s);
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try link.sendOp(.{ wire.MAGIC, 0, @intCast(active >> 20), @intCast(calib), 0, 0, 0, 0 });
    log.info("[dist] worker ready ({d} MiB resident, {d} us per {d}-row pass)\n", .{ active >> 20, calib, CALIB_ROWS });
    _ = try measureLink(link, io);

    var bcast: ?h3.AttnBroadcast = null;
    defer if (bcast) |*b| b.deinit();
    var steps: u64 = 0;
    while (true) {
        const m = try link.recvOp();
        switch (m[0]) {
            wire.OP_STOP => break,
            wire.OP_RELEASE => return true,
            wire.OP_H3_JOB => {
                const turbo = m[1] != 0;
                if (turbo != turbo_bound) {
                    const lora = try std.fs.path.join(allocator, &.{ dir, "turbo_lora.safetensors" });
                    defer allocator.free(lora);
                    const paths: []const []const u8 = if (turbo) &.{lora} else &.{};
                    const scales: []const f32 = if (turbo) &.{1.0} else &.{};
                    try h3.bindLoras(allocator, &stack, &model, paths, scales, turbo);
                    turbo_bound = turbo;
                }
                if (bcast) |*b| b.deinit();
                bcast = if (m[2] != 0) try h3.AttnBroadcast.init(allocator, model.blocks.len) else null;
            },
            wire.OP_H3_END => {
                if (bcast) |*b| b.deinit();
                bcast = null;
                _ = mlx.mlx_clear_cache();
                log.info("[dist] worker: video job done after {d} steps\n", .{steps});
                steps = 0;
            },
            wire.OP_H3_STEP => {
                const rows: u32 = @intCast(m[1]);
                const cut: u32 = @intCast(m[2]);
                const n1: c_int = @intCast(rows - cut);
                const dt = model.dtype;
                var h = try link.recvNow(&.{ n1, @intCast(model.cfg.hidden_size) }, dt, 0);
                defer _ = mlx.mlx_array_free(h);
                const t_emb = try link.recvNow(&.{ m[5], m[6] }, dt, 0);
                defer _ = mlx.mlx_array_free(t_emb);
                const runs_a = try link.recvNow(&.{m[3] * 3}, .int32, 0);
                defer _ = mlx.mlx_array_free(runs_a);
                var rope: h3.RopeTables = .{
                    .cos = try link.recvNow(&.{ 1, n1, 1, m[7] }, dt, 0),
                    .sin = try link.recvNow(&.{ 1, n1, 1, m[7] }, dt, 0),
                };
                defer rope.deinit();
                const flat = (mlx.mlx_array_data_int32(runs_a) orelse return error.DistBadMessage)[0..@intCast(m[3] * 3)];
                const runs = try allocator.alloc(h3.ModRun, @intCast(m[3]));
                defer allocator.free(runs);
                for (runs, 0..) |*r, i| r.* = .{ .start = @intCast(flat[i * 3]), .end = @intCast(flat[i * 3 + 1]), .mod_row = @intCast(flat[i * 3 + 2]) };
                const split: RowSplit = .{ .link = link, .rank = 1, .rows = rows, .cut = cut };
                model.split = &split;
                defer model.split = null;
                try model.runBlocks(&h, t_emb, .{ .ctx = null }, runs, rope, null, if (bcast) |*b| b else null, m[4] != 0, s);
                try mlx.check(mlx.mlx_array_eval(h));
                try link.sendNow(h, 0);
                steps += 1;
            },
            else => return error.DistBadMessage,
        }
    }
    return false;
}

fn cat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

/// `x[lo:hi]` on `axis`, contiguous.
fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const step: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[axis] = lo;
    stop[axis] = hi;
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_slice(&v, x, &start, sh.len, &stop, sh.len, &step, sh.len, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, v, false, s));
    return o;
}

/// `g[i:i+1, :, :n]` of a gathered `[4, H, nmax, hd]`.
fn slice4(g: mlx.mlx_array, i: c_int, n: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(g);
    const start = [_]c_int{ i, 0, 0, 0 };
    const stop = [_]c_int{ i + 1, sh[1], n, sh[3] };
    const step: [4]c_int = @splat(1);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, g, &start, 4, &stop, 4, &step, 4, s));
    return o;
}

test "localRuns clips each run to one Mac's rows and renumbers them" {
    const a = std.testing.allocator;
    const runs = [_]h3.ModRun{ .{ .start = 0, .end = 10, .mod_row = 0 }, .{ .start = 10, .end = 100, .mod_row = 3 }, .{ .start = 100, .end = 120, .mod_row = 6 } };
    const lo = try localRuns(a, &runs, 0, 50);
    defer a.free(lo);
    try std.testing.expectEqualSlices(h3.ModRun, &.{ .{ .start = 0, .end = 10, .mod_row = 0 }, .{ .start = 10, .end = 50, .mod_row = 3 } }, lo);
    const hi = try localRuns(a, &runs, 50, 120);
    defer a.free(hi);
    try std.testing.expectEqualSlices(h3.ModRun, &.{ .{ .start = 0, .end = 50, .mod_row = 3 }, .{ .start = 50, .end = 70, .mod_row = 6 } }, hi);
}

test "chooseCut: twin Macs split evenly, a much slower worker or a small canvas stays alone" {
    const kv: u64 = 2 * 56 * 128 * 2;
    // Equal speed, a big canvas, a 9 GB/s link: half each.
    try std.testing.expectEqual(@as(?u32, 15000), chooseCut(30000, 600_000, 600_000, 9000, kv, 50));
    // The same Macs over a slow link: the swaps eat the gain.
    try std.testing.expectEqual(@as(?u32, null), chooseCut(30000, 600_000, 600_000, 2000, kv, 50));
    // The worker at a third of this Mac's speed: it saves too little.
    try std.testing.expectEqual(@as(?u32, null), chooseCut(30000, 600_000, 1_800_000, 9000, kv, 50));
    // A faster worker takes more of the rows.
    try std.testing.expect(chooseCut(30000, 900_000, 600_000, 9000, kv, 50).? < 15000);
    // A tiny canvas: the per-block sync outweighs the work.
    try std.testing.expectEqual(@as(?u32, null), chooseCut(64, 600_000, 600_000, 9000, kv, 50));
}
