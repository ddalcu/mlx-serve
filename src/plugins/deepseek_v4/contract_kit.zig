//! The DeepSeek family's kernel-contract test type: one generic `Case(D)` per kernel, where `D` declares the launch's
//! geometry, the inputs a gate asks for (`fill`), the output count and the float64 formula (`compute`, which with an
//! accumulator set also simulates gate 4's sums). Optional `D.check` states a rule of its own (quantizers, top-k).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");

pub const spec = kc.spec;
pub const metal = kc.metal;
pub const gates = kc.gates;
pub const Shape = spec.Shape;
pub const Fill = spec.Fill;
pub const Accum = spec.Accum;
pub const Order = spec.Order;

pub const nan = std.math.nan(f64);
pub const inf = std.math.inf(f64);

/// One input as the kernel binds it: dtype, shape and its values (already rounded to the dtype).
pub const In = struct { dt: mlx.mlx_dtype, dims: [4]c_int = @splat(0), nd: usize, v: []f64 };

/// How a `compute` sums: float64 (`acc` null) or gate 4's simulated accumulator in an order.
pub const Sum = struct {
    acc: ?Accum = null,
    order: Order = .sequential,

    /// `t`'s terms summed as asked.
    pub fn of(s: Sum, t: []const f64) f64 {
        if (s.acc) |a| return spec.sumIn(a, s.order, t);
        var r: f64 = 0;
        for (t) |v| r += v;
        return r;
    }
};

pub fn Case(comptime D: type) type {
    return struct {
        const Self = @This();

        rig: metal.Rig,
        launch: *const metal.Launch,
        p: D.Params,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        ins: [16]In = undefined,
        n_in: usize = 0,
        args: [16]mlx.mlx_array = undefined,
        /// A stated rule's finding, returned by `check`.
        msg: [256]u8 = undefined,

        pub fn init(a: std.mem.Allocator, launch: *const metal.Launch, p: D.Params) Self {
            return .{ .rig = metal.Rig.init(a), .launch = launch, .p = p, .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
            return D.geometry(self.p, shape);
        }

        /// A new input of `dims` as `dt`, its values to fill (zeroed); bound in the order added.
        pub fn add(self: *Self, dt: mlx.mlx_dtype, dims: []const usize) ![]f64 {
            var n: usize = 1;
            var x = In{ .dt = dt, .nd = dims.len, .v = &.{} };
            for (dims, 0..) |d, i| {
                n *= d;
                x.dims[i] = @intCast(d);
            }
            x.v = try self.arena.allocator().alloc(f64, n);
            @memset(x.v, 0);
            self.ins[self.n_in] = x;
            self.n_in += 1;
            return x.v;
        }

        /// Input `i`'s values.
        pub fn in(self: *const Self, i: usize) []f64 {
            return self.ins[i].v;
        }

        pub fn scratch(self: *const Self) std.mem.Allocator {
            return self.arena.child_allocator;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: Fill, seed: u64) !bool {
            if (D.geometry(self.p, shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            self.n_in = 0;
            var prng = std.Random.DefaultPrng.init(seed);
            try D.fill(self, fill, prng.random());
            self.rig.reset();
            for (self.ins[0..self.n_in], 0..) |x, i| {
                for (x.v) |*v| v.* = kc.host.rt(x.dt, v.*);
                self.args[i] = try self.rig.input(x.dt, x.dims[0..x.nd], x.v);
            }
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, D.geometry(self.p, self.shape).?, self.args[0..self.n_in]);
        }

        pub fn count(self: *const Self) usize {
            return D.count(self.p, self.shape);
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            @memset(mag, 0);
            try D.compute(self, want, mag, .{});
        }

        pub fn simulate(self: *const Self, acc: Accum, order: Order, out: []f64) !void {
            const mag = try self.scratch().alloc(f64, out.len);
            defer self.scratch().free(mag);
            try D.compute(self, out, mag, .{ .acc = acc, .order = order });
        }

        pub fn replaced(self: *Self) ![]const u8 {
            if (comptime @hasDecl(D, "replaced")) return D.replaced(self) else unreachable;
        }

        pub fn check(self: *Self, at: spec.Gate) !?[]const u8 {
            if (comptime @hasDecl(D, "check")) return D.check(self, at) else unreachable;
        }

        /// The last run's outputs read as `e`, into `out`.
        pub fn outputs(self: *Self, e: spec.Elem, out: []f64) void {
            for (out, 0..) |*o, i| o.* = spec.read(e, self.rig.outputBytes(), i);
        }
    };
}

/// Registers `D` at `p`: the harness's hooks, with `check` only where `D` states a rule for these params.
pub fn add(set: *kc.Set, comptime D: type, launch: *const metal.Launch, p: D.Params, s: spec.Spec) !void {
    const T = Case(D);
    const V = struct {
        const full = blk: {
            var v = gates.hooks(T);
            if (!@hasDecl(D, "replaced")) v.replaced = null;
            break :blk v;
        };
        const plain = blk: {
            var v = full;
            v.check = null;
            break :blk v;
        };
        fn free(a: std.mem.Allocator, ctx: *anyopaque) void {
            const c: *T = @ptrCast(@alignCast(ctx));
            c.deinit();
            a.destroy(c);
        }
    };
    const checks = @hasDecl(D, "check") and (!@hasDecl(D, "checks") or D.checks(p));
    const c = try set.a.create(T);
    errdefer set.a.destroy(c);
    c.* = .init(set.a, launch, p);
    try set.frees.ensureUnusedCapacity(set.a, 1);
    try set.kernels.append(set.a, .{ .spec = s, .ctx = c, .vt = if (checks) &V.full else &V.plain });
    set.frees.appendAssumeCapacity(V.free);
}

/// Uniform values in [lo, hi).
pub fn uniform(rng: std.Random, out: []f64, lo: f64, hi: f64) void {
    for (out) |*v| v.* = lo + (hi - lo) * rng.float(f64);
}

/// Places NaN, +Inf and -Inf in `v` (a [rows, k] row-major input): row 0 a NaN, row 1 +Inf, row 2 -Inf, row 3 both
/// infinities; rows past 3 stay finite.
pub fn plantRows(v: []f64, rows: usize, k: usize) void {
    const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
    const put = [_]f64{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, val| if (p[0] < rows) {
        v[p[0] * k + p[1]] = val;
    };
}

/// IEEE max that propagates NaN (MLX's `maximum`), unlike Metal's `fmax`.
pub fn maxN(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return nan;
    return @max(a, b);
}

pub fn minN(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return nan;
    return @min(a, b);
}

/// Compares the last run's outputs to `want` with per-output allowances `allow` (`got == want` passes, classes must
/// match); null when every output passes, else a description in `buf`.
pub fn within(buf: []u8, got: []const f64, want: []const f64, allow: []const f64) ?[]const u8 {
    for (got, want, allow, 0..) |g, w, l, i| {
        if (spec.classOf(g) != spec.classOf(w)) return std.fmt.bufPrint(buf, "output {d} is {e} where float64 gives {e}", .{ i, g, w }) catch "class differs";
        if (g == w or spec.classOf(w) != .finite) continue;
        if (!(@abs(g - w) <= l)) return std.fmt.bufPrint(buf, "output {d} is {e} where float64 gives {e} (allowed {e})", .{ i, g, w, l }) catch "outside";
    }
    return null;
}

// ── tests ──

test "deepseek kernel kit: a float64 sum and a simulated one" {
    const t = [_]f64{ 1, 0x1p-30, -1 };
    try std.testing.expectEqual(@as(f64, 0x1p-30), (Sum{}).of(&t));
    try std.testing.expectEqual(@as(f64, 0), (Sum{ .acc = .f32 }).of(&t));
}
