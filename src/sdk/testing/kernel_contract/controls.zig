//! The planted controls: tiny row kernels each built to break one gate (a maximum by `fmax` that drops a NaN, a sum
//! by atomic adds, a sum that writes its input or a guard byte, a tail that drops its last input, a grid narrower
//! than the rows it commits, a dot summed in half, a launch past M1's thread cap, 36 KiB of threadgroup memory).
//! Each must fail its gate; one that passes fails the run. `Row` is also the smallest example of a test type.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const spec = @import("spec.zig");
const metal = @import("metal.zig");
const Set = @import("set.zig").Set;

const Shape = spec.Shape;

/// A row op over x [rows, k] f32 (and w [k] for a dot), one f32 output a row.
pub const Op = enum { max, sum, dot };

pub const Row = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    op: Op,
    dispatch: *const fn (Shape) metal.Geometry,
    shape: Shape = undefined,
    x: []f64 = &.{},
    w: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    pub fn init(a: std.mem.Allocator, launch: *const metal.Launch, op: Op, dispatch: *const fn (Shape) metal.Geometry) Row {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .op = op, .dispatch = dispatch };
    }

    pub fn deinit(self: *Row) void {
        self.rig.deinit();
        self.rig.a.free(self.x);
        self.rig.a.free(self.w);
    }

    pub fn geometry(self: *const Row, shape: Shape) ?metal.Geometry {
        return self.dispatch(shape);
    }

    pub fn prepare(self: *Row, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
        self.x = try a.alloc(f64, shape.rows * shape.k);
        self.w = try a.alloc(f64, shape.k);
        fillRows(self.x, self.w, shape, fill, seed);
        self.rig.reset();
        const k: c_int = @intCast(shape.k);
        self.args[0] = try self.rig.input(.float32, &.{ @intCast(shape.rows), k }, self.x);
        self.args[1] = try self.rig.input(.float32, &.{k}, self.w);
        return true;
    }

    pub fn run(self: *Row) !void {
        try self.rig.launch(self.launch, self.dispatch(self.shape), &self.args);
    }

    pub fn count(self: *const Row) usize {
        return self.shape.rows;
    }

    /// Row `r`'s summed terms into `t`.
    fn terms(self: *const Row, r: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, 0..) |*o, i| o.* = self.x[r * k + i] * (if (self.op == .dot) self.w[i] else 1);
    }

    pub fn reference(self: *const Row, want: []f64, mag: []f64) !void {
        const a = self.rig.a;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        for (want, mag, 0..) |*y, *m, r| {
            self.terms(r, t);
            y.* = if (self.op == .max) -std.math.inf(f64) else 0;
            m.* = 0;
            for (t) |v| switch (self.op) {
                .max => y.* = if (std.math.isNan(v) or std.math.isNan(y.*)) std.math.nan(f64) else @max(y.*, v),
                else => {
                    y.* += v;
                    m.* += @abs(v);
                },
            };
        }
    }

    pub fn simulate(self: *const Row, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        if (self.op == .max) return error.NoSums;
        const t = try self.rig.a.alloc(f64, self.shape.k);
        defer self.rig.a.free(t);
        for (out, 0..) |*y, r| {
            self.terms(r, t);
            y.* = spec.sumIn(acc, order, t);
        }
    }
};

/// Gate inputs for x [rows, k] and w [k], f32-exact: spread magnitudes; NaN and both infinities at known places
/// (row 3 holds both infinities); or gate 4's mirrored integers (big partial sums, a small result).
pub fn fillRows(x: []f64, w: []f64, shape: Shape, fill: spec.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const k = shape.k;
    for (w) |*v| v.* = spec.roundTo(.f32, rng.float(f64) * 2 - 1);
    for (x) |*v| v.* = spec.roundTo(.f32, (rng.float(f64) * 2 - 1) * std.math.pow(f64, 2, @floatFromInt(rng.intRangeAtMost(i32, -8, 8))));
    switch (fill) {
        .random => {},
        .nonfinite => {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                x[p[0] * k + p[1]] = v;
            };
        },
        .stress => {
            const h = k / 2;
            for (0..shape.rows) |r| {
                const row = x[r * k ..][0..k];
                for (0..h) |i| {
                    const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                    row[i] = v;
                    row[h + i] = v + @as(f64, if (i < 8) 1 else 0);
                }
                if (k % 2 == 1) row[k - 1] = 0;
            }
            for (0..h) |i| {
                const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                w[i] = v;
                w[h + i] = -v;
            }
            if (k % 2 == 1) w[k - 1] = 0;
        },
    }
}

/// A control's launch: its file, and what gate 5 reads (atomic outputs, M1/M2 limit, threadgroup bytes).
const Opts = struct { atomic: bool = false, m: u32 = 1024, tg_bytes: u32 = 0, offline: bool = true };

fn kernel(comptime name: []const u8, comptime o: Opts) metal.Launch {
    return .{
        .name = "control_" ++ name,
        .inputs = &.{ "x", "w" },
        .outputs = &.{"y"},
        .source = @embedFile("controls/" ++ name ++ ".metal"),
        .atomic_outputs = o.atomic,
        .tg_bytes = o.tg_bytes,
        .m1 = o.m,
        .m2 = o.m,
        .offline = o.offline,
    };
}

const fmax_max = kernel("row_max_fmax", .{});
const atomic_sum = kernel("row_sum_atomic", .{ .atomic = true });
// The full unroll over K takes registers: M1/M2 build these for 928 threads (`offline.zig`).
const writes_input = kernel("row_sum_writes_input", .{ .m = 928 });
const writes_guard = kernel("row_sum_writes_guard", .{});
const tail_drop = kernel("row_sum_tail_drop", .{});
const narrow_grid = kernel("row_sum_narrow_grid", .{});
const half_dot = kernel("row_dot_half", .{ .m = 928 });
const tg36k = kernel("row_copy_tg36k", .{ .tg_bytes = 36 * 1024, .offline = false, .m = 0 });

/// One thread a row.
fn perRow(shape: Shape) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ @intCast(shape.rows), 1, 1 }, .tg = .{ @intCast(@min(shape.rows, 32)), 1, 1 } };
    g.out(&.{@intCast(shape.rows)}, .float32);
    g.int("K", shape.k);
    return g;
}

/// One thread an input, adding into its row's output (which starts at zero).
fn perInput(shape: Shape) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.k), 1, 1 }, .tg = .{ 64, 1, 1 }, .poison = 0 };
    g.out(&.{@intCast(shape.rows)}, .float32);
    g.int("K", shape.k);
    return g;
}

/// One thread every four rows, rounded down: a fifth row is never written.
fn perFourRows(shape: Shape) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ @intCast(@max(1, shape.rows / 4)), 1, 1 }, .tg = .{ 1, 1, 1 } };
    g.out(&.{@intCast(shape.rows)}, .float32);
    g.int("K", shape.k);
    g.int("ROWS", shape.rows);
    return g;
}

/// One group of a thread a row, 1024 rows: past the 928 threads M1/M2 build the unrolled row sum for.
fn wideGroup(shape: Shape) metal.Geometry {
    var g = perRow(shape);
    g.tg = .{ @intCast(shape.rows), 1, 1 };
    return g;
}

fn chainDepth(shape: Shape) usize {
    return shape.k;
}

fn rowSpec(comptime name: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = "control " ++ name,
        .accum = .f32,
        .depth = chainDepth,
        .out = .f32,
        .tiles = .{ .{ .size = 4 }, .{}, .{ .size = 32 } },
        .served = served,
    };
}

const plain: []const Shape = &.{.{ .rows = 8, .n = 1, .k = 64 }};

pub fn register(set: *Set) !void {
    const a = set.a;
    try set.add(Row, Row.init(a, &fmax_max, .max, perRow), rowSpec("row max by fmax", plain), .nonfinite);
    try set.add(Row, Row.init(a, &atomic_sum, .sum, perInput), rowSpec("row sum by atomic adds", &.{.{ .rows = 4, .n = 1, .k = 16384 }}), .repeat);
    try set.add(Row, Row.init(a, &writes_input, .sum, perRow), rowSpec("row sum writing its input", plain), .repeat);
    var guard = rowSpec("in-place row sum writing a guard byte", plain);
    guard.in_place = true;
    try set.add(Row, Row.init(a, &writes_guard, .sum, perRow), guard, .repeat);
    try set.add(Row, Row.init(a, &tail_drop, .sum, perRow), rowSpec("row sum dropping its tail's last input", plain), .shapes);
    try set.add(Row, Row.init(a, &narrow_grid, .sum, perFourRows), rowSpec("row sum on a grid narrower than its rows", plain), .shapes);
    try set.add(Row, Row.init(a, &half_dot, .dot, perRow), rowSpec("row dot summed in half", &.{.{ .rows = 4, .n = 1, .k = 4096 }}), .sums);
    var flat = rowSpec("row sum at 1024 threads where M1/M2 build 928", &.{.{ .rows = 1024, .n = 1, .k = 64 }});
    flat.tiles = .{ .{}, .{}, .{} };
    try set.add(Row, Row.init(a, &writes_input, .sum, wideGroup), flat, .limits);
    flat.name = "control row copy through 36 KiB of threadgroup memory";
    try set.add(Row, Row.init(a, &tg36k, .sum, perRow), flat, .limits);
}
