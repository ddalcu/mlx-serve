//! The sampling area's kernel contract: keyed Gumbel-max sampling (`msv_keyed_sample`). Its output is a token id,
//! so it checks itself (`check`): against a float64 replay of the rule (temperature, top_k, the top_p nucleus,
//! min_p, the Gumbel draw at the row's position) that accepts only the choices the device's f32 sums and fast-math
//! logs can make near a nucleus edge or a score tie; and its stated non-finite rule. Its f32 sums (the normalizer,
//! the nucleus) feed a choice, not an output, so gate 4 does not apply. Served on every Apple GPU (M1 and up).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const ks = @import("keyed_sample.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

/// Candidates a row sorts (the wrapper's `CANDIDATES`), and the near window (`NEAR`).
const C = 1024;
const NEAR: f32 = 20;

/// Threadgroup bytes: fsh, fsh2, ush (32 each), hist[256], st[4], ck/ci[C], two atomics.
const launch: metal.Launch = .{
    .name = "msv_keyed_sample",
    .inputs = &.{ "L", "seeds", "positions", "cfg", "kcap" },
    .outputs = &.{"TOK"},
    .source = kernelSource("kernels/sampling/keyed_sample.metal"),
    .header = kernelSource("kernels/sampling/keyed_sample_header.metal"),
    .tg_bytes = 3 * 32 * 4 + 256 * 4 + 16 + 2 * C * 4 + 8,
    .m1 = 1024,
    .m2 = 1024,
};

const NAN_RULE = "a NaN or -Inf logit is never drawn; a row holding +Inf draws a +Inf token";

/// A shape's rows are the rows sampled (positions base, base + 1, ...), `k` the vocabulary.
const Sample = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    p: ks.Params,
    dt: mlx.mlx_dtype,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    logits: []f64 = &.{},
    base: u32 = 0,
    args: [5]mlx.mlx_array = undefined,
    msg: [160]u8 = undefined,

    fn init(a: std.mem.Allocator, p: ks.Params, dt: mlx.mlx_dtype) Sample {
        return .{ .rig = metal.Rig.init(a), .launch = &launch, .p = p, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *Sample) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const Sample, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.rows > 64) return null;
        var g = metal.Geometry{ .grid = .{ 1024 * @as(c_int, @intCast(shape.rows)), 1, 1 }, .tg = .{ 1024, 1, 1 } };
        g.out(&.{@intCast(shape.rows)}, .uint32);
        g.int("V", shape.k);
        g.int("C", C);
        return g;
    }

    pub fn prepare(self: *Sample, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const v = shape.k;
        // Logits shaped like a model's: a broad body, a few hundred tokens near the top.
        self.logits = try a.alloc(f64, rows * v);
        for (self.logits) |*x| x.* = host.rt(self.dt, rng.floatNorm(f64) * 2 - 4 + @as(f64, if (rng.uintLessThan(u32, 1000) == 0) 8 else 0));
        if (fill == .nonfinite) host.plant(self.logits, rows, v);
        self.base = rng.uintLessThan(u32, 4096);
        const rig = &self.rig;
        rig.reset();
        const seeds = try a.alloc(f64, 2 * rows);
        const pos = try a.alloc(f64, rows);
        for (0..rows) |r| {
            seeds[2 * r] = @floatFromInt(@as(u32, @truncate(self.p.seed)));
            seeds[2 * r + 1] = @floatFromInt(@as(u32, @truncate(self.p.seed >> 32)));
            pos[r] = @floatFromInt(self.base + r);
        }
        const cfg = [_]f64{ 1.0 / @max(self.p.temperature, 1e-6), self.p.top_p, NEAR, self.p.min_p };
        var cfg32: [4]f64 = undefined;
        for (&cfg32, cfg) |*o, x| o.* = host.rt(.float32, x);
        self.args = .{
            try rig.input(self.dt, &.{ @intCast(rows), @intCast(v) }, self.logits),
            try rig.input(.uint32, &.{@intCast(2 * rows)}, seeds),
            try rig.input(.uint32, &.{@intCast(rows)}, pos),
            try rig.input(.float32, &.{4}, &cfg32),
            try rig.input(.uint32, &.{1}, &[_]f64{@floatFromInt(self.p.top_k)}),
        };
        return true;
    }

    pub fn run(self: *Sample) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Sample) usize {
        return self.shape.rows;
    }

    /// No float64 values to compare: the token is checked by `check`.
    pub fn reference(_: *const Sample, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn token(self: *Sample, r: usize) u32 {
        return std.mem.readInt(u32, self.rig.outputBytes()[4 * r ..][0..4], .little);
    }

    fn fail(self: *Sample, comptime fmt: []const u8, args: anytype) ?[]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "check failed";
    }

    pub fn check(self: *Sample, gate: spec.Gate) !?[]const u8 {
        const v = self.shape.k;
        for (0..self.shape.rows) |r| {
            const row = self.logits[r * v ..][0..v];
            const t = self.token(r);
            if (t >= v) return self.fail("row {d} drew id {d} past the vocabulary", .{ r, t });
            if (gate == .nonfinite) {
                const x = row[t];
                if (std.math.isNan(x) or x == -std.math.inf(f64)) return self.fail("row {d} drew a {d} logit", .{ r, x });
                const has_inf = for (row) |y| {
                    if (y == std.math.inf(f64)) break true;
                } else false;
                if (has_inf and x != std.math.inf(f64)) return self.fail("row {d} holds +Inf but drew {d}", .{ r, x });
                continue;
            }
            if (try self.why(r, row, t)) |msg| return msg;
        }
        return null;
    }

    /// Null when `t` is a choice the rule allows row `r` within the device's arithmetic, else why not.
    fn why(self: *Sample, r: usize, row: []const f64, t: u32) !?[]const u8 {
        const a = self.rig.a;
        const p = self.p;
        const inv_t: f32 = @floatCast(host.rt(.float32, 1.0 / @max(p.temperature, 1e-6)));
        const vals = try a.alloc(f64, row.len);
        defer a.free(vals);
        var m = -std.math.inf(f64);
        for (vals, row) |*o, x| {
            o.* = @as(f32, @floatCast(x)) * inv_t;
            m = @max(m, o.*);
        }
        const ids = try a.alloc(u32, row.len);
        defer a.free(ids);
        for (ids, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, ids, vals, struct {
            fn lt(x: []const f64, i: u32, j: u32) bool {
                return x[i] > x[j] or (x[i] == x[j] and i < j);
            }
        }.lt);
        const cap: usize = if (p.top_k == 0 or p.top_k > C) C else p.top_k;
        const n: usize = @min(@min(row.len, C), cap);
        var z: f64 = 0;
        for (vals) |x| z += @exp(x - m);
        var norm = z;
        if (p.top_k != 0) {
            norm = 0;
            for (ids[0..n]) |i| norm += @exp(vals[i] - m);
        }
        // The nucleus and min_p cuts, each as early and as late as the device's sums and logs can put them.
        const eps_p = 1e-4;
        const eps_v = 1e-5 * (1 + @abs(m));
        var lo = n;
        var hi = n;
        if (p.top_p > 0 and p.top_p < 1) {
            var cum: f64 = 0;
            var got_lo = false;
            for (ids[0..n], 0..) |i, j| {
                cum += @exp(vals[i] - m) / norm;
                if (!got_lo and cum >= p.top_p - eps_p) {
                    lo = j + 1;
                    got_lo = true;
                }
                if (cum >= p.top_p + eps_p) {
                    hi = j + 1;
                    break;
                }
            }
        }
        if (p.min_p > 0) {
            const floor_v = m + @log(@as(f64, p.min_p));
            for (ids[0..lo], 0..) |i, j| if (vals[i] < floor_v + eps_v) {
                lo = j;
                break;
            };
            for (ids[0..hi], 0..) |i, j| if (vals[i] < floor_v - eps_v) {
                hi = j;
                break;
            };
        }
        const rank = std.mem.indexOfScalar(u32, ids[0..n], t) orelse return self.fail("row {d} drew {d}, outside the {d} candidates", .{ r, t, n });
        if (rank >= hi) return self.fail("row {d} drew rank {d}, past the nucleus (at most {d})", .{ r, rank, hi });
        const pos: u64 = self.base + r;
        const mine = vals[t] + ks.gumbel(p.seed, pos, t);
        for (ids[0..@max(lo, rank + 1)]) |i| {
            const other = vals[i] + ks.gumbel(p.seed, pos, i);
            if (other > mine + 1e-5 * (1 + @abs(other))) return self.fail("row {d} drew {d} (score {d:.6}) where {d} scores {d:.6}", .{ r, t, mine, i, other });
        }
        return null;
    }

    /// The shipped wrapper on the same rows.
    fn wrapper(self: *Sample) ![]const u8 {
        var pos: [64]u32 = undefined;
        for (pos[0..self.shape.rows], 0..) |*o, r| o.* = self.base + @as(u32, @intCast(r));
        const tok = try ks.sample(self.args[0], self.p, pos[0..self.shape.rows], self.rig.stream());
        defer _ = mlx.mlx_array_free(tok);
        return self.rig.bytesOf(&.{tok});
    }
};

fn depthNone(_: Shape) usize {
    return 0;
}

fn sampleSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{ .name = name, .accum = .none, .depth = depthNone, .out = .f32, .nan = .{ .stated = NAN_RULE }, .served = served };
}

const qwen35: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 248320 }, .{ .rows = 4, .n = 1, .k = 248320 } };
const qwen3: []const Shape = &.{ .{ .rows = 1, .n = 1, .k = 151936 }, .{ .rows = 3, .n = 1, .k = 151936 } };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Sample, .init(a, .{ .seed = 1234, .temperature = 1.0, .top_p = 0.95, .top_k = 20 }, .bfloat16), sampleSpec("keyed_sample bf16 (T 1, top_k 20, top_p 0.95)", qwen35), null);
    try set.add(Sample, .init(a, .{ .seed = 99, .temperature = 0.7, .top_p = 0.9, .top_k = 0 }, .float32), sampleSpec("keyed_sample f32 (T 0.7, top_p 0.9)", qwen35), null);
    try set.add(Sample, .init(a, .{ .seed = 7, .temperature = 2.0, .top_p = 1.0, .top_k = 0 }, .float32), sampleSpec("keyed_sample f32 (T 2, no cut: the radix path)", qwen3), null);
    try set.add(Sample, .init(a, .{ .seed = 5, .temperature = 0.8, .top_p = 1.0, .top_k = 40, .min_p = 0.05 }, .bfloat16), sampleSpec("keyed_sample bf16 (top_k 40, min_p 0.05)", qwen3), null);
}

// ── tests ──

test "kernel contract sampling: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const t: *Sample = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try t.prepare(k.spec.served[1], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
}
