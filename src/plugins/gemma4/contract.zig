//! Gemma 4's kernel contract: the fused PLE tail at decode (`gelu_ple.zig`), bit-equal to the GELU table lookup,
//! multiply and 4-bit group-64 `quantized_matmul` it replaces. Listed in `kernel_contract/registry.zig`.
//! Shapes: `rows` counts threadgroups of 8 output rows (the projection's hidden width / 8); `k` is the per-layer
//! width, 256 (the only one the wrapper takes).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const gelu_ple = @import("gelu_ple.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const PD = 256;
/// Output rows a threadgroup writes (two simdgroups of four).
const ROWS_PER_TG = 8;
/// Elements a lane reads, and lanes a group of scales covers.
const PER_LANE = 8;
const LANES_PER_GROUP = 8;
const TABLE = 1 << 16;

/// Threadgroup bytes: `act[PD]` in T.
const launch = metal.Launch{
    .name = "mlxserve_gelu_ple_proj",
    .inputs = &.{ "g", "ple", "gtab", "pw", "ps", "pb" },
    .outputs = &.{"y"},
    .source = gelu_ple.GELU_PLE_PROJ_SOURCE,
    .tg_bytes = 2 * PD,
    .m1 = 1024,
    .m2 = 1024,
};

const Proj = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    g: []f64 = &.{},
    ple: []f64 = &.{},
    gtab: []f64 = &.{},
    /// Packed nibbles as u32 words, `[hid, PD / 8]`; scales and biases `[hid, PD / 64]`.
    pw: []f64 = &.{},
    ps: []f64 = &.{},
    pb: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Proj {
        return .{ .rig = metal.Rig.init(a), .launch = &launch, .arena = .init(a) };
    }

    pub fn deinit(self: *Proj) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn hid(shape: Shape) c_int {
        return @intCast(shape.rows * ROWS_PER_TG);
    }

    pub fn geometry(_: *const Proj, shape: Shape) ?metal.Geometry {
        if (shape.k != PD or shape.rows == 0) return null;
        const h = hid(shape);
        var g = metal.Geometry{ .grid = .{ 64 * @divTrunc(h, ROWS_PER_TG), 1, 1 }, .tg = .{ 64, 1, 1 } };
        g.out(&.{ 1, 1, h }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("HID", h);
        g.int("PD", @as(c_int, PD));
        return g;
    }

    pub fn prepare(self: *Proj, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows: usize = @intCast(hid(shape));
        const groups = PD / 64;
        const bf = struct {
            fn vals(al: std.mem.Allocator, r: std.Random, n: usize, lo: f64, hi: f64) ![]f64 {
                const out = try al.alloc(f64, n);
                for (out) |*v| v.* = spec.roundTo(.bf16, lo + (hi - lo) * r.float(f64));
                return out;
            }
        };
        self.g = try bf.vals(a, rng, PD, -4, 4);
        self.ple = try bf.vals(a, rng, PD, -2, 2);
        self.gtab = try bf.vals(a, rng, TABLE, -1, 1);
        self.pw = try a.alloc(f64, rows * PD / 8);
        for (self.pw) |*w| w.* = @floatFromInt(rng.int(u32));
        self.ps = try bf.vals(a, rng, rows * groups, 0.01, 0.1);
        self.pb = try bf.vals(a, rng, rows * groups, -0.1, 0.1);
        switch (fill) {
            .random => {},
            // Row 0 a NaN scale, row 1 a +Inf scale, row 2 a -Inf bias, row 3 both infinities; the rest finite.
            .nonfinite => {
                self.ps[0 * groups + 0] = std.math.nan(f64);
                self.ps[1 * groups + 1] = std.math.inf(f64);
                self.pb[2 * groups + 2] = -std.math.inf(f64);
                self.ps[3 * groups + 3] = std.math.inf(f64);
                self.pb[3 * groups + 3] = -std.math.inf(f64);
            },
            // Gate 4: every weight nibble 1, scale 1, bias 0, the table 1, so a row is the sum of `ple`:
            // +4096, 1, -4096, 1 repeating. fp32 sums them exactly in any order; fp16 and bf16 drop every 1 beside a
            // 4096 (sequential and pairwise) or beside a lane of eight (strided), so they miss the 128.
            .stress => {
                @memset(self.gtab, 1);
                for (self.ple, 0..) |*v, i| v.* = switch (i % 4) {
                    0 => 4096,
                    2 => -4096,
                    else => 1,
                };
                @memset(self.pw, 0x11111111);
                @memset(self.ps, 1);
                @memset(self.pb, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const h = hid(shape);
        const vec: []const c_int = &.{ 1, 1, PD };
        self.args = .{
            try rig.input(.bfloat16, vec, self.g),
            try rig.input(.bfloat16, vec, self.ple),
            try rig.input(.bfloat16, &.{TABLE}, self.gtab),
            try rig.input(.uint32, &.{ h, PD / 8 }, self.pw),
            try rig.input(.bfloat16, &.{ h, groups }, self.ps),
            try rig.input(.bfloat16, &.{ h, groups }, self.pb),
        };
        return true;
    }

    pub fn run(self: *Proj) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Proj) usize {
        return @intCast(hid(self.shape));
    }

    /// `gtab[bits(g)] * ple`, rounded to bf16 (the kernel stores it so before the projection).
    fn act(self: *const Proj, i: usize) f64 {
        return spec.roundTo(.bf16, self.gtab[spec.toBf16(@floatCast(self.g[i]))] * self.ple[i]);
    }

    fn nibble(self: *const Proj, r: usize, i: usize) f64 {
        const word: u32 = @intFromFloat(self.pw[r * (PD / 8) + i / 8]);
        return @floatFromInt((word >> @intCast(4 * (i % 8))) & 0xf);
    }

    /// Row `row` in float64, summed as the kernel associates it (a lane's `scale * sum(x q) + bias * sum(x)`, then
    /// over lanes), so an infinite scale or bias meets the same signs.
    fn row(self: *const Proj, r: usize, mag: *f64) f64 {
        var total: f64 = 0;
        mag.* = 0;
        for (0..PD / PER_LANE) |lane| {
            const grp = r * (PD / 64) + lane / LANES_PER_GROUP;
            var xq: f64 = 0;
            var xs: f64 = 0;
            var axq: f64 = 0;
            var axs: f64 = 0;
            for (0..PER_LANE) |j| {
                const i = lane * PER_LANE + j;
                const x = self.act(i);
                xq += x * self.nibble(r, i);
                xs += x;
                axq += @abs(x * self.nibble(r, i));
                axs += @abs(x);
            }
            total += self.ps[grp] * xq + self.pb[grp] * xs;
            mag.* += @abs(self.ps[grp]) * axq + @abs(self.pb[grp]) * axs;
        }
        return total;
    }

    pub fn reference(self: *const Proj, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, r| w.* = self.row(r, m);
    }

    /// Each output's terms `x (s q + b)` summed in `acc` in `order`.
    pub fn simulate(self: *const Proj, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        var t: [PD]f64 = undefined;
        for (out, 0..) |*o, r| {
            for (&t, 0..) |*v, i| {
                const grp = r * (PD / 64) + i / 64;
                v.* = self.act(i) * (self.ps[grp] * self.nibble(r, i) + self.pb[grp]);
            }
            o.* = spec.sumIn(acc, order, &t);
        }
    }

    /// The chain it replaces on the same inputs: table lookup by bits, multiply, `quantized_matmul`.
    pub fn replaced(self: *Proj) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const x = self.args;
        const bits = try g.call(mlx.mlx_view, .{ x[0], mlx.mlx_dtype.uint16 });
        const gated = try g.mul(try g.take(x[2], bits, 0), x[1]);
        return self.rig.bytesOf(&.{try g.qmm(gated, x[3], x[4], x[5], 64, 4)});
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Proj) ![]const u8 {
        const x = self.args;
        const y = (try gelu_ple.fusedGeluPleProj(self.rig.stream(), x[0], x[1], x[2], x[3], x[4], x[5], .{ .bits = 4, .group_size = 64 })) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// A lane's chain of 8 products, then the simdgroup's tree of 32.
fn depth(_: Shape) usize {
    return PER_LANE + 5;
}

pub fn register(set: *kc.Set) !void {
    try set.add(Proj, .init(set.a), .{
        .name = "gelu_ple_proj bf16 (4-bit g64, PD 256)",
        .accum = .f32,
        .depth = depth,
        // f32 roundings after the lane sums: scale * accum, bias * sum, their add.
        .ops = 3,
        .out = .bf16,
        .replaces = "gelu table take -> multiply -> quantized_matmul (4-bit g64)",
        // E2B's and E4B's hidden widths, in threadgroups of 8 rows.
        .served = &.{ .{ .rows = 1536 / ROWS_PER_TG, .n = 1, .k = PD }, .{ .rows = 2560 / ROWS_PER_TG, .n = 1, .k = PD } },
    }, null);
}

// ── tests ──

test "kernel contract gemma4: the declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    fused_act.swiglu_fused_override = true;
    defer fused_act.swiglu_fused_override = null;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const p: *Proj = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try p.prepare(k.spec.served[0], .random, 7));
        try p.run();
        const ours = try a.dupe(u8, p.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
    }
}
