//! Portable streamed DIG GEMMs: guarded launches of the served texts against the host trellis decoder.
//! Repeated 16x16 code tiles keep the full-width reference bounded; each activation row remains independent.
const std = @import("std");
const bo = @import("build_options");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const xk = if (bo.mlx_stream) @import("mlx_stream").exl3_kernels else struct {};
const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;
const header = @embedFile("stream/kernels/exl3/portable_prefill.metal");

const Dig = struct {
    rig: metal.Rig,
    arena: std.heap.ArenaAllocator,
    work: std.heap.ArenaAllocator,
    reg: *xk.Registry,
    kernel: xk.Kernel,
    launch: *metal.Launch,
    shape: Shape = undefined,
    inputs: [6]mlx.mlx_array = undefined,
    n_inputs: usize = 0,
    xs: [2][]f64 = .{ &.{}, &.{} },
    weights: [2][256]f64 = undefined,
    n_ops: usize,
    kd: usize,
    nd: usize,
    bits: usize,
    bm: usize,
    fused: bool,

    fn init(a: std.mem.Allocator, kernel: xk.Kernel) !Dig {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const p = arena.allocator();
        const reg = try p.create(xk.Registry);
        reg.* = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, null);
        errdefer reg.deinit();
        const e = reg.get(kernel);
        const gate = e.outputs.len == 2;
        const n_ops: usize = if (gate) 2 else 1;
        const fused = e.inputs.len == 4;
        const launch = try p.create(metal.Launch);
        const names = try p.alloc([*:0]const u8, e.inputs.len);
        for (names, e.inputs) |*n, input| n.* = input.name.ptr;
        const outs = try p.alloc([*:0]const u8, e.outputs.len);
        for (outs, e.outputs) |*n, output| n.* = output.name.ptr;
        const lut = kernel == .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut;
        // Lowest native-code equality bounds over the three targets in each M1/M2 family.
        // These are the offline contract's measurements, not a runtime pipeline-property query.
        const offline_limit: u32 = if (fused) 192 else if (e.inputs[e.inputs.len - 1].domain.bm == 64) 96 else 128;
        launch.* = .{
            .name = e.mlx_name,
            .source = e.source,
            .header = try std.mem.concatWithSentinel(p, u8, &.{ reg.header(e), "\n", header }, 0),
            .inputs = names,
            .outputs = outs,
            .contiguous = e.ensure_row_contiguous,
            .tg_bytes = (if (fused) @as(u32, 128) else 64) * 72 * 2 + (if (lut) @as(u32, 2048) else 0),
            .m1 = offline_limit,
            .m2 = offline_limit,
        };
        return .{
            .rig = .init(a),
            .arena = arena,
            .work = .init(a),
            .reg = reg,
            .kernel = kernel,
            .launch = launch,
            .n_ops = n_ops,
            .kd = e.inputs[0].shape[2].m,
            .nd = e.outputs[0].shape[1].m,
            .bits = e.inputs[n_ops].shape[3].m / 16,
            .bm = e.inputs[e.inputs.len - 1].domain.bm,
            .fused = fused,
        };
    }

    pub fn deinit(self: *Dig) void {
        self.rig.deinit();
        self.work.deinit();
        self.reg.deinit();
        self.arena.deinit();
    }

    fn vars(self: *const Dig, rows: usize) xk.Vars {
        var v: xk.Vars = .initFill(0);
        const e = self.reg.get(self.kernel);
        v.set(.rows, rows);
        v.set(.cap, 1);
        v.set(.experts, 1);
        v.set(.tgs, ((rows + self.bm - 1) / self.bm) * e.inputs[e.inputs.len - 1].domain.tiles);
        return v;
    }

    pub fn geometry(self: *const Dig, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n != self.nd or shape.k != self.kd) return null;
        const cfg = xk.launchFor(self.reg.get(self.kernel), &self.vars(shape.rows), null) catch return null;
        var g: metal.Geometry = .{ .grid = undefined, .tg = undefined };
        for (&g.grid, cfg.grid) |*out, value| out.* = @intCast(value);
        for (&g.tg, cfg.threadgroup) |*out, value| out.* = @intCast(value);
        for (0..cfg.n_out) |i| g.out(cfg.out_shapes[i][0..cfg.out_ranks[i]], cfg.out_dtypes[i]);
        return g;
    }

    fn arg(self: *Dig, value: mlx.mlx_array) void {
        self.inputs[self.n_inputs] = value;
        self.n_inputs += 1;
    }

    pub fn prepare(self: *Dig, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.rig.reset();
        _ = self.work.reset(.free_all);
        const a = self.work.allocator();
        self.shape = shape;
        self.n_inputs = 0;
        var prng: std.Random.DefaultPrng = .init(seed);
        const rng = prng.random();
        for (0..self.n_ops) |op| {
            self.xs[op] = try a.alloc(f64, shape.rows * self.kd);
            for (self.xs[op], 0..) |*v, i| {
                v.* = switch (fill) {
                    .random, .nonfinite => @as(f64, @floatFromInt(rng.intRangeAtMost(i16, -32, 32))) / 16,
                    // The zero trellis state is -221/64. These inputs make every product and
                    // every partial sum integral and below 2^24, in all three reference orders.
                    .stress => if (i % self.kd == 0) 32768 else 64,
                };
            }
            if (fill == .nonfinite) {
                self.xs[op][0] = std.math.nan(f64);
                if (shape.rows > 1) self.xs[op][self.kd] = std.math.inf(f64);
                if (shape.rows > 2) self.xs[op][2 * self.kd] = -std.math.inf(f64);
                if (shape.rows > 3) {
                    self.xs[op][3 * self.kd] = std.math.inf(f64);
                    self.xs[op][3 * self.kd + 1] = -std.math.inf(f64);
                }
            }
            self.arg(try self.rig.input(.float16, &.{ @intCast(shape.rows), 1, @intCast(self.kd) }, self.xs[op]));
        }
        const table = try a.create([65536]u16);
        xk.mul1Table(table);
        for (0..self.n_ops) |op| {
            var tile: [48]i16 = undefined;
            const words = tile[0 .. 16 * self.bits];
            for (words) |*w| w.* = if (fill == .stress) 0 else rng.int(i16);
            var decoded: [256]u16 = undefined;
            xk.reconstruct(words, 1, 1, self.bits, table, &decoded, null);
            for (&self.weights[op], decoded) |*w, bits| w.* = @as(f16, @bitCast(bits));
            const codes = try a.alloc(i16, self.kd * self.nd / 256 * words.len);
            var at: usize = 0;
            while (at < codes.len) : (at += words.len) @memcpy(codes[at..][0..words.len], words);
            self.arg(try self.rig.inputRaw(.int16, &.{ 1, @intCast(self.kd / 16), @intCast(self.nd / 16), @intCast(words.len) }, std.mem.sliceAsBytes(codes)));
        }
        if (self.fused) {
            const rout = try a.alloc(f64, self.nd);
            @memset(rout, 1);
            self.arg(try self.rig.input(.float16, &.{ 1, @intCast(self.nd) }, rout));
        }
        var tbl: [80]i32 = @splat(0);
        @memset(tbl[48..64], std.math.maxInt(i32));
        tbl[32] = @intCast(shape.rows);
        tbl[48] = 0;
        self.arg(try self.rig.inputRaw(.int32, &.{80}, std.mem.sliceAsBytes(&tbl)));
        return true;
    }

    pub fn run(self: *Dig) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.inputs[0..self.n_inputs]);
    }

    pub fn count(self: *const Dig) usize {
        return self.shape.rows * self.nd * self.n_ops;
    }

    fn compute(self: *Dig, out: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const a = self.work.allocator();
        const terms = try a.alloc(f64, self.kd);
        defer a.free(terms);
        for (0..self.n_ops) |op| for (0..self.shape.rows) |row| {
            var z: [128]f64 = undefined;
            var m: [128]f64 = undefined;
            for (0..16) |col| {
                var val: f64 = 0;
                var abs_sum: f64 = 0;
                for (terms, 0..) |*term, k| {
                    term.* = self.xs[op][row * self.kd + k] * self.weights[op][k % 16 * 16 + col];
                    val += term.*;
                    abs_sum += @abs(term.*);
                }
                z[col] = if (acc) |p| spec.sumIn(p, order, terms) else val;
                m[col] = abs_sum;
            }
            for (16..128) |col| {
                z[col] = z[col % 16];
                m[col] = m[col % 16];
            }
            if (self.fused) {
                var half: usize = 1;
                while (half < 128) : (half *= 2) {
                    var block: usize = 0;
                    while (block < 128) : (block += 2 * half) for (block..block + half) |j| {
                        const x = z[j];
                        const y = z[j + half];
                        z[j] = x + y;
                        z[j + half] = x - y;
                        const both = m[j] + m[j + half];
                        m[j] = both;
                        m[j + half] = both;
                    };
                }
                const scale: f64 = @as(f32, @bitCast(@as(u32, 1035273459)));
                for (&z, &m) |*v, *w| {
                    v.* *= scale;
                    w.* *= scale;
                }
            }
            const base = (op * self.shape.rows + row) * self.nd;
            for (out[base..][0..self.nd], 0..) |*v, col| {
                v.* = z[col % 128];
                if (mag) |magnitude| magnitude[base + col] = m[col % 128];
            }
        };
    }

    pub fn reference(self: *Dig, out: []f64, mag: []f64) !void {
        try self.compute(out, mag, null, .sequential);
    }
    pub fn simulate(self: *Dig, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.compute(out, null, acc, order);
    }
};

fn depth(shape: Shape) usize {
    return shape.k + 7;
}

pub fn register(set: *kc.Set) !void {
    if (comptime !bo.mlx_stream) return;
    inline for (xk.portable_prefill_kernels) |kernel| {
        var d = try Dig.init(set.a, kernel);
        errdefer d.deinit();
        const a = d.arena.allocator();
        const shapes = try a.alloc(Shape, 3);
        for (shapes, [_]usize{ 1, 129, 257 }) |*shape, rows| shape.* = .{ .rows = rows, .n = d.nd, .k = d.kd };
        const name = try std.fmt.allocPrint(a, "portable {s}", .{@tagName(kernel)});
        try set.add(Dig, d, .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = if (d.fused) 2 else 0,
            .out = .f32,
            .tiles = .{ .{ .size = d.bm }, .{}, .{} },
            .served = shapes,
            .sums = .{},
            .known = .{
                .gate = .limits,
                .finding = "offline native-code equality bounds are below the fixed launch size; the donor ran on M2, but migrated M1/M2 runtime pipeline limits are unverified and model loading remains refused",
            },
        }, null);
    }
}

test "kernel contract portable DIG: nine served launches have five-gate registrations" {
    if (comptime !bo.mlx_stream) return error.SkipZigTest;
    var set = kc.Set.init(std.testing.allocator);
    defer set.deinit();
    try register(&set);
    try std.testing.expectEqual(@as(usize, 9), set.kernels.items.len);
    for (set.kernels.items) |k| {
        try std.testing.expect(k.vt.simulate != null);
        try std.testing.expect(k.vt.geometry != null);
        try std.testing.expect(k.vt.launch != null);
    }
}

test "kernel contract portable DIG: served wrapper and guarded launch agree byte for byte" {
    if (comptime !bo.mlx_stream) return error.SkipZigTest;
    if (mlx.noGpuBackend() or @import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    for (set.kernels.items) |k| {
        const d: *Dig = @ptrCast(@alignCast(k.ctx));
        const shape: Shape = .{ .rows = 33, .n = d.nd, .k = d.kd };
        try std.testing.expect(try d.prepare(shape, .random, 19));
        try d.run();
        const want = d.rig.outputBytes();
        var bound = try d.reg.bind(d.rig.stream(), null);
        defer bound.deinit();
        const cfg = try xk.launchFor(d.reg.get(d.kernel), &d.vars(shape.rows), null);
        var out: [2]mlx.mlx_array = @splat(.{});
        try bound.apply(d.kernel, d.inputs[0..d.n_inputs], &cfg, out[0..cfg.n_out]);
        defer for (out[0..cfg.n_out]) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var at: usize = 0;
        for (out[0..cfg.n_out]) |x| {
            try mlx.check(mlx.mlx_array_eval(x));
            const n = mlx.mlx_array_size(x) * 4;
            const ptr = mlx.mlx_array_data_uint8(x) orelse return error.NoData;
            try std.testing.expectEqualSlices(u8, want[at..][0..n], ptr[0..n]);
            at += n;
        }
        try std.testing.expectEqual(want.len, at);
    }
}

test "kernel contract portable DIG: numerical gates and source capture" {
    if (comptime !bo.mlx_stream) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var tally: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &tally);
    std.debug.print("portable DIG numerical gates: {d}/{d} pass, {d} problems\n", .{ tally.passed, tally.kernels, tally.problems() });
    if (@import("../../../env.zig").get(.MLX_SERVE_KERNEL_CONTRACT_OFFLINE)) |dir| {
        try std.Io.Dir.cwd().createDirPath(std.testing.io, std.mem.span(dir));
        for (set.kernels.items) |k| {
            const d: *Dig = @ptrCast(@alignCast(k.ctx));
            try std.testing.expect(try d.prepare(.{ .rows = 129, .n = d.nd, .k = d.kd }, .random, 19));
            var src: std.ArrayList(u8) = .empty;
            defer src.deinit(a);
            d.rig.capture = &src;
            defer d.rig.capture = null;
            try d.run();
            const path = try std.fmt.allocPrint(a, "{s}/{s}.metal", .{ std.mem.span(dir), @tagName(d.kernel) });
            defer a.free(path);
            try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = src.items });
            var scratch = std.heap.ArenaAllocator.init(a);
            defer scratch.deinit();
            const measured = try @import("../../testing/kernel_contract/offline.zig").measureSource(scratch.allocator(), std.testing.io, std.mem.span(dir), src.items);
            std.debug.print("portable DIG limits {s}: M1 {d}, M2 {d}\n", .{ @tagName(d.kernel), measured[0], measured[1] });
        }
    }
    try std.testing.expectEqual(@as(usize, 0), tally.problems());
}
