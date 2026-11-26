//! Kernel contract registrations (Gates 1-5 + shipped-wrapper parity) for Inkling 2's vision Metal kernels
//! (`vision_kernels.zig`): QKV bias + 2-D RoPE, NAX bidirectional attention, bias + layer-scaled residual + LayerNorm,
//! and bias + exact erf-GELU.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const vk = @import("vision_kernels.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

extern "c" fn erf(f64) f64;

fn elemOf(dt: mlx.mlx_dtype) spec.Elem {
    return switch (dt) {
        .bfloat16 => .bf16,
        .float16 => .f16,
        else => .f32,
    };
}

fn dtName(comptime dt: mlx.mlx_dtype) []const u8 {
    return comptime switch (dt) {
        .bfloat16 => "bf16",
        .float16 => "f16",
        else => "f32",
    };
}

fn none(_: Shape) usize {
    return 0;
}

fn rowLen(shape: Shape) usize {
    return shape.k;
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

// ── 1. vision_qkv_rope.metal ──

const qkv_rope_launch = metal.Launch{
    .name = "msv_inkling2_vis_qkv_rope",
    .inputs = &.{ "QKV", "B", "COS", "SIN" },
    .outputs = &.{ "OQ", "OK", "OV" },
    .source = vk.QKV_ROPE_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

fn qkvPartCount(shape: Shape) usize {
    return shape.rows * shape.n * shape.k;
}

fn VisQkvRope(comptime dt: mlx.mlx_dtype) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &qkv_rope_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        qkv: []f64 = &.{},
        bias: []f64 = &.{},
        cos: []f64 = &.{},
        sin: []f64 = &.{},
        args: [4]mlx.mlx_array = undefined,

        const Self = @This();

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 2 != 0) return null;
            const total: c_int = cint(3 * shape.n * (shape.k / 2));
            const tg = @min(total, @as(c_int, 256));
            const gx = @divTrunc(total + tg - 1, tg) * tg;
            var g = metal.Geometry{ .grid = .{ gx, cint(shape.rows), 1 }, .tg = .{ tg, 1, 1 } };
            for (0..3) |_| g.out(&.{ 1, cint(shape.n), cint(shape.rows), cint(shape.k) }, dt);
            g.dtype("T", dt);
            g.int("H", shape.n);
            g.int("D", shape.k);
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const width = 3 * shape.n * shape.k;
            self.qkv = try host.uniform(a, rng, shape.rows * width, -3, 3, dt);
            self.bias = try host.uniform(a, rng, width, -1, 1, dt);
            self.cos = try a.alloc(f64, shape.rows * shape.k);
            self.sin = try a.alloc(f64, shape.rows * shape.k);
            for (0..shape.rows) |r| {
                for (0..shape.k / 2) |p| {
                    const ang = if (r == 0) 0.0 else 6.3 * rng.float(f64);
                    const c = spec.roundTo(.f32, @cos(ang));
                    const s = spec.roundTo(.f32, @sin(ang));
                    self.cos[r * shape.k + 2 * p] = c;
                    self.cos[r * shape.k + 2 * p + 1] = c;
                    self.sin[r * shape.k + 2 * p] = s;
                    self.sin[r * shape.k + 2 * p + 1] = s;
                }
            }
            if (fill == .nonfinite) host.plant(self.qkv, shape.rows, width);
            self.rig.reset();
            self.args = .{
                try self.rig.input(dt, &.{ cint(shape.rows), cint(width) }, self.qkv),
                try self.rig.input(dt, &.{cint(width)}, self.bias),
                try self.rig.input(.float32, &.{ cint(shape.rows), cint(shape.k) }, self.cos),
                try self.rig.input(.float32, &.{ cint(shape.rows), cint(shape.k) }, self.sin),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return 3 * qkvPartCount(self.shape);
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            const l = self.shape.rows;
            const h = self.shape.n;
            const d = self.shape.k;
            const elem = elemOf(dt);
            const part = l * h * d;
            for (0..3) |which| {
                for (0..h) |head| {
                    const slot = which * h + head;
                    for (0..l) |r| {
                        for (0..d / 2) |p| {
                            const col = slot * d + 2 * p;
                            const src = r * 3 * h * d + col;
                            const v0 = spec.roundTo(elem, self.qkv[src] + self.bias[col]);
                            const v1 = spec.roundTo(elem, self.qkv[src + 1] + self.bias[col + 1]);
                            const dst = which * part + (head * l + r) * d + 2 * p;
                            if (which == 2) {
                                want[dst] = v0;
                                want[dst + 1] = v1;
                                mag[dst] = @abs(v0);
                                mag[dst + 1] = @abs(v1);
                            } else {
                                const tab = r * d + 2 * p;
                                const c0 = self.cos[tab];
                                const c1 = self.cos[tab + 1];
                                const s0 = self.sin[tab];
                                const s1 = self.sin[tab + 1];
                                want[dst] = v0 * c0 + (-v1) * s0;
                                want[dst + 1] = v1 * c1 + v0 * s1;
                                mag[dst] = @abs(v0 * c0) + @abs(v1 * s0);
                                mag[dst + 1] = @abs(v1 * c1) + @abs(v0 * s1);
                            }
                        }
                    }
                }
            }
        }

        pub fn replaced(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const l = cint(self.shape.rows);
            const h = cint(self.shape.n);
            const d = cint(self.shape.k);
            const hd_total = h * d;
            const added = try g.add(self.args[0], self.args[1]);
            const cos3 = try g.reshape(self.args[2], &.{ l, 1, d });
            const sin3 = try g.reshape(self.args[3], &.{ l, 1, d });
            const perm = [_]c_int{ 1, 0, 2 };
            const rot = struct {
                fn apply(gr: *ops.Graph, x: ops.Array, c: ops.Array, s: ops.Array) !ops.Array {
                    const sh = ops.shape(x);
                    const f = try gr.astype(x, .float32);
                    const pairs = try gr.reshape(f, &.{ sh[0], sh[1], @divExact(sh[2], 2), 2 });
                    const turned = try gr.reshape(try gr.concat(&.{ try gr.neg(try gr.sliceAxis(pairs, 3, 1, 2)), try gr.sliceAxis(pairs, 3, 0, 1) }, 3), sh);
                    return gr.astype(try gr.add(try gr.mul(f, c), try gr.mul(turned, s)), dt);
                }
            }.apply;
            const q = try g.expandDims(try g.transpose(try rot(&g, try g.reshape(try g.sliceAxis(added, 1, 0, hd_total), &.{ l, h, d }), cos3, sin3), &perm), 0);
            const k = try g.expandDims(try g.transpose(try rot(&g, try g.reshape(try g.sliceAxis(added, 1, hd_total, 2 * hd_total), &.{ l, h, d }), cos3, sin3), &perm), 0);
            const v = try g.expandDims(try g.transpose(try g.reshape(try g.sliceAxis(added, 1, 2 * hd_total, 3 * hd_total), &.{ l, h, d }), &perm), 0);
            return self.rig.bytesOf(&.{ q, k, v });
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const x = self.args;
            const outs = try vk.qkvRope(&g, x[0], x[1], x[2], x[3], cint(self.shape.n), cint(self.shape.k));
            return self.rig.bytesOf(&outs);
        }
    };
}

// ── 2. vision_attn_nax.metal ──

const attn_nax_launch = metal.Launch{
    .name = "msv_inkling2_vis_attn_nax",
    .inputs = &.{ "Q", "K", "V", "scale" },
    .outputs = &.{"O"},
    .source = vk.ATTN_NAX_SOURCE,
    .header = vk.ATTN_NAX_HEADER,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
    .offline = false,
    .min_gen = 17,
};

fn VisAttnNax(comptime dt: mlx.mlx_dtype) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &attn_nax_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        q: []f64 = &.{},
        k: []f64 = &.{},
        v: []f64 = &.{},
        args: [4]mlx.mlx_array = undefined,

        const Self = @This();
        const SCALE: f32 = 0.125;
        const BK: usize = 32;
        const LOG2E: f64 = 1.4426950408889634;
        const vis_soft = attn.Soft{ .base2 = true, .p_unit = 0x1p-11, .p_abs = 0x1p-25 };

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.n == 0 or shape.k != 64) return null;
            var g = metal.Geometry{ .grid = .{ @divTrunc(cint(shape.rows) + 63, 64) * 128, cint(shape.n), 1 }, .tg = .{ 128, 1, 1 } };
            g.out(&.{ cint(shape.rows), cint(shape.n * 64) }, dt);
            g.dtype("T", dt);
            g.int("H", shape.n);
            g.int("BK", @as(c_int, 32));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const n = shape.n * shape.rows * 64;
            self.q = try host.uniform(a, rng, n, -1.5, 1.5, dt);
            self.k = try host.uniform(a, rng, n, -1.5, 1.5, dt);
            self.v = try host.uniform(a, rng, n, -1.5, 1.5, dt);
            if (fill == .nonfinite) host.plant(self.q, shape.n * shape.rows, 64);
            self.rig.reset();
            const sh = [_]c_int{ 1, cint(shape.n), cint(shape.rows), 64 };
            self.args = .{
                try self.rig.input(dt, &sh, self.q),
                try self.rig.input(dt, &sh, self.k),
                try self.rig.input(dt, &sh, self.v),
                try self.rig.scalar(SCALE),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return self.shape.rows * self.shape.n * 64;
        }

        fn emit(self: *const Self, want: []f64, mag: ?[]f64, sim: attn.Sim) !void {
            const a = self.rig.a;
            const l = self.shape.rows;
            const h = self.shape.n;
            const d: usize = 64;
            const keys = try a.alloc(attn.Key, l);
            defer a.free(keys);
            const ends = try a.alloc(usize, (l + BK - 1) / BK);
            defer a.free(ends);
            const terms = try a.alloc(f64, d);
            defer a.free(terms);
            const o = try a.alloc(f64, d);
            defer a.free(o);
            const eo = try a.alloc(f64, d);
            defer a.free(eo);
            const zero_err: [d]f64 = @splat(0);
            const qs: f64 = @as(f32, @floatCast(@as(f64, SCALE) * LOG2E));
            var nb: usize = 0;
            var k0: usize = 0;
            while (k0 < l) : (k0 += BK) {
                ends[nb] = @min(k0 + BK, l);
                nb += 1;
            }
            const start: attn.Init = .{ .m = -std.math.inf(f64), .l = 0 };
            for (0..h) |head| {
                const h_off = head * l * d;
                for (0..l) |qi| {
                    const q_row = self.q[h_off + qi * d ..][0..d];
                    for (0..l) |ki| {
                        const k_row = self.k[h_off + ki * d ..][0..d];
                        const v_row = self.v[h_off + ki * d ..][0..d];
                        const dp = attn.dot(q_row, k_row, qs, d, 1, sim, terms);
                        keys[ki] = .{ .s = dp.v, .e = dp.e, .v = v_row, .ve = &zero_err };
                    }
                    const st = attn.online(vis_soft, start, keys, ends[0..nb], o, eo);
                    attn.normalize(st, o, eo);
                    const dst = qi * (h * d) + head * d;
                    for (0..d) |c| {
                        want[dst + c] = o[c];
                        if (mag) |m| m[dst + c] = if (std.math.isFinite(eo[c])) host.allowance(eo[c]) else 0;
                    }
                }
            }
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            try self.emit(want, mag, null);
        }

        pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            try self.emit(out, null, .{ .acc = acc, .order = order });
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const x = self.args;
            return self.rig.bytesOf(&.{try vk.attnNax(&g, x[0], x[1], x[2], x[3])});
        }
    };
}

// ── 3. vision_res_norm.metal ──

const res_norm_launch = metal.Launch{
    .name = "msv_inkling2_vis_res_norm",
    .inputs = &.{ "X", "Y", "B", "SC", "NW", "NB", "eps" },
    .outputs = &.{ "OUT", "NORM" },
    .source = vk.RES_NORM_SOURCE,
    .tg_bytes = 64,
    .m1 = 1024,
    .m2 = 1024,
};

fn resNormCount(shape: Shape) usize {
    return shape.rows * shape.k;
}

fn VisResNorm(comptime dt: mlx.mlx_dtype, comptime do_norm: bool) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &res_norm_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        x: []f64 = &.{},
        y: []f64 = &.{},
        b: []f64 = &.{},
        sc: []f64 = &.{},
        nw: []f64 = &.{},
        nb: []f64 = &.{},
        args: [7]mlx.mlx_array = undefined,

        const Self = @This();
        const EPS: f32 = 1e-5;

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.k == 0) return null;
            const tg = vk.resNormTg(cint(shape.k));
            var g = metal.Geometry{ .grid = .{ tg, cint(shape.rows), 1 }, .tg = .{ tg, 1, 1 } };
            g.out(&.{ cint(shape.rows), cint(shape.k) }, dt);
            if (do_norm) {
                g.out(&.{ cint(shape.rows), cint(shape.k) }, dt);
            } else {
                g.out(&.{1}, dt);
                g.unread |= 1 << 1;
            }
            g.dtype("T", dt);
            g.int("D", shape.k);
            g.int("TG", tg);
            g.int("DO_NORM", @as(c_int, @intFromBool(do_norm)));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const n = shape.rows * shape.k;
            const d = shape.k;
            self.x = try host.uniform(a, rng, n, -3, 3, dt);
            self.y = try host.uniform(a, rng, n, -3, 3, dt);
            self.b = try host.uniform(a, rng, d, -0.5, 0.5, dt);
            self.sc = try host.uniform(a, rng, d, 0.2, 1.5, dt);
            self.nw = try host.uniform(a, rng, d, 0.5, 1.5, dt);
            self.nb = try host.uniform(a, rng, d, -0.5, 0.5, dt);
            switch (fill) {
                .random => {},
                .nonfinite => host.plant(self.x, shape.rows, d),
                .stress => {
                    @memset(self.y, 0);
                    @memset(self.b, 0);
                    @memset(self.sc, 1);
                    @memset(self.nw, 1);
                    @memset(self.nb, 0);
                    const cut = (5 * d) / 8;
                    for (0..shape.rows) |r| {
                        for (0..d) |c| {
                            self.x[r * d + c] = if (c < cut) 64.5 else 63.5;
                        }
                    }
                },
            }
            self.rig.reset();
            self.args = .{
                try self.rig.input(dt, &.{ cint(shape.rows), cint(d) }, self.x),
                try self.rig.input(dt, &.{ cint(shape.rows), cint(d) }, self.y),
                try self.rig.input(dt, &.{cint(d)}, self.b),
                try self.rig.input(dt, &.{cint(d)}, self.sc),
                try self.rig.input(dt, &.{cint(d)}, self.nw),
                try self.rig.input(dt, &.{cint(d)}, self.nb),
                try self.rig.scalar(EPS),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return (if (do_norm) @as(usize, 2) else 1) * resNormCount(self.shape);
        }

        fn emit(self: *const Self, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
            const elem = elemOf(dt);
            const rows = self.shape.rows;
            const d = self.shape.k;
            const n = rows * d;
            const terms = try self.rig.a.alloc(f64, d);
            defer self.rig.a.free(terms);
            for (0..rows) |r| {
                const base = r * d;
                for (0..d) |c| {
                    const yb = spec.roundTo(elem, self.y[base + c] + self.b[c]);
                    const ys = spec.roundTo(elem, yb * self.sc[c]);
                    const m = spec.roundTo(elem, self.x[base + c] + ys);
                    want[base + c] = m;
                    if (mag) |mg| mg[base + c] = @abs(self.x[base + c]) + @abs(ys);
                    terms[c] = m;
                }
                if (!do_norm) continue;
                const sum = if (acc) |ac| spec.sumIn(ac, order, terms) else host.sum(terms);
                const mean = sum / @as(f64, @floatFromInt(d));
                const sq = try self.rig.a.alloc(f64, d);
                defer self.rig.a.free(sq);
                for (0..d) |c| {
                    const diff = terms[c] - mean;
                    sq[c] = diff * diff;
                }
                const var_sum = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
                const inv = 1.0 / @sqrt(var_sum / @as(f64, @floatFromInt(d)) + EPS);
                for (0..d) |c| {
                    const normed = (terms[c] - mean) * inv * self.nw[c] + self.nb[c];
                    want[n + base + c] = normed;
                    if (mag) |mg| mg[n + base + c] = @max(@abs(normed), @abs(self.nw[c]) + @abs(self.nb[c]));
                }
            }
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            try self.emit(want, mag, null, .sequential);
        }

        pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            try self.emit(out, null, acc, order);
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const x = self.args;
            const outs = try vk.resNorm(&g, x[0], x[1], x[2], x[3], x[4], x[5], x[6], do_norm);
            return if (do_norm) self.rig.bytesOf(&outs) else self.rig.bytesOf(&.{outs[0]});
        }
    };
}

// ── 4. vision_bias_gelu.metal ──

const bias_gelu_launch = metal.Launch{
    .name = "msv_inkling2_vis_bias_gelu",
    .inputs = &.{ "Y", "B" },
    .outputs = &.{"O"},
    .source = vk.BIAS_GELU_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

fn VisBiasGelu(comptime dt: mlx.mlx_dtype, comptime has_bias: bool) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = &bias_gelu_launch,
        arena: std.heap.ArenaAllocator,
        shape: Shape = undefined,
        y: []f64 = &.{},
        b: []f64 = &.{},
        args: [2]mlx.mlx_array = undefined,

        const Self = @This();

        fn init(a: std.mem.Allocator) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn geometry(_: *const Self, shape: Shape) ?metal.Geometry {
            if (shape.rows == 0 or shape.k == 0) return null;
            const quads = @divTrunc(cint(shape.k) + 3, 4);
            const tg = @min(quads, @as(c_int, 256));
            const gx = @divTrunc(quads + tg - 1, tg) * tg;
            var g = metal.Geometry{ .grid = .{ gx, cint(shape.rows), 1 }, .tg = .{ tg, 1, 1 } };
            g.out(&.{ cint(shape.rows), cint(shape.k) }, dt);
            g.dtype("T", dt);
            g.int("HAS_BIAS", @as(c_int, @intFromBool(has_bias)));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const n = shape.rows * shape.k;
            self.y = try host.uniform(a, rng, n, -6, 6, dt);
            self.b = try host.uniform(a, rng, if (has_bias) shape.k else 1, -0.5, 0.5, dt);
            if (!has_bias) self.b[0] = 0;
            if (fill == .nonfinite) host.plant(self.y, shape.rows, shape.k);
            self.rig.reset();
            self.args = .{
                try self.rig.input(dt, &.{ cint(shape.rows), cint(shape.k) }, self.y),
                try self.rig.input(dt, &.{cint(if (has_bias) shape.k else 1)}, self.b),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            return self.shape.rows * self.shape.k;
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            const elem = elemOf(dt);
            const rows = self.shape.rows;
            const m = self.shape.k;
            for (0..rows) |r| {
                for (0..m) |c| {
                    const idx = r * m + c;
                    const x = if (has_bias) spec.roundTo(elem, self.y[idx] + self.b[c]) else self.y[idx];
                    const w = 0.5 * x * (1.0 + erf(x * std.math.sqrt1_2));
                    want[idx] = w;
                    mag[idx] = @max(@abs(w), @abs(x) * 0.5);
                }
            }
        }

        fn wrapper(self: *Self) ![]const u8 {
            var g = ops.Graph.init(self.rig.a, self.rig.stream());
            defer g.deinit();
            const x = self.args;
            return self.rig.bytesOf(&.{try vk.biasGelu(&g, x[0], if (has_bias) x[1] else .{})});
        }
    };
}

const vis_dtypes = [_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 };
const nax_dtypes = [_]mlx.mlx_dtype{ .bfloat16, .float16 };

pub fn register(set: *kc.Set) !void {
    inline for (vis_dtypes) |dt| {
        const elem = comptime elemOf(dt);
        const tn = comptime dtName(dt);
        try set.add(VisQkvRope(dt), .init(set.a), .{
            .name = "inkling2 vision_qkv_rope " ++ tn,
            .accum = .none,
            .depth = none,
            .ops = 4,
            .mids = 1,
            .out = elem,
            .outs = &.{
                .{ .elem = elem, .count = qkvPartCount },
                .{ .elem = elem, .count = qkvPartCount },
                .{ .elem = elem, .count = qkvPartCount },
            },
            .replaces = "qkv bias add + 2-D rotary + head transpose",
            .served = &.{ .{ .rows = 25, .n = 4, .k = 64 }, .{ .rows = 145, .n = 16, .k = 64 } },
        }, null);
    }
    const attn_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "softmax exp leaves no sum exact: stress checks fp32 sums within float64 bound" };
    inline for (nax_dtypes) |dt| {
        const tn = comptime dtName(dt);
        try set.add(VisAttnNax(dt), .init(set.a), .{
            .name = "inkling2 vision_attn_nax " ++ tn,
            .accum = .f32,
            .depth = host.depthOne,
            .out = comptime elemOf(dt),
            .tiles = .{ .{ .size = 64, .min = 1 }, .{}, .{} },
            .served = &.{ .{ .rows = 25, .n = 4, .k = 64 }, .{ .rows = 145, .n = 16, .k = 64 } },
            .sums = attn_sums,
            .nax = true,
        }, null);
    }
    inline for (vis_dtypes) |dt| {
        const elem = comptime elemOf(dt);
        const tn = comptime dtName(dt);
        try set.add(VisResNorm(dt, true), .init(set.a), .{
            .name = "inkling2 vision_res_norm " ++ tn ++ " (norm)",
            .accum = .f32,
            .depth = rowLen,
            .ops = 8,
            .mids = 3,
            .out = elem,
            .outs = &.{
                .{ .elem = elem, .count = resNormCount },
                .{ .elem = elem, .count = resNormCount },
            },
            .served = &.{ .{ .rows = 16, .n = 1, .k = 128 }, .{ .rows = 25, .n = 1, .k = 1024 }, .{ .rows = 8, .n = 1, .k = 2048 } },
            .sums = .{ .f16 = 16, .bf16 = 16, .exact = false },
        }, null);
        try set.add(VisResNorm(dt, false), .init(set.a), .{
            .name = "inkling2 vision_res_norm " ++ tn ++ " (residual only)",
            .accum = .none,
            .depth = none,
            .ops = 3,
            .mids = 2,
            .out = elem,
            .served = &.{ .{ .rows = 16, .n = 1, .k = 128 }, .{ .rows = 25, .n = 1, .k = 1024 }, .{ .rows = 8, .n = 1, .k = 2048 } },
        }, null);
    }
    inline for (vis_dtypes) |dt| {
        const tn = comptime dtName(dt);
        try set.add(VisBiasGelu(dt, true), .init(set.a), .{
            .name = "inkling2 vision_bias_gelu " ++ tn ++ " (biased)",
            .accum = .none,
            .depth = none,
            .ops = 16,
            .mids = 1,
            .out = elemOf(dt),
            .served = &.{ .{ .rows = 16, .n = 1, .k = 130 }, .{ .rows = 25, .n = 1, .k = 4096 } },
        }, null);
        try set.add(VisBiasGelu(dt, false), .init(set.a), .{
            .name = "inkling2 vision_bias_gelu " ++ tn ++ " (unbiased)",
            .accum = .none,
            .depth = none,
            .ops = 16,
            .out = elemOf(dt),
            .served = &.{ .{ .rows = 16, .n = 1, .k = 130 }, .{ .rows = 25, .n = 1, .k = 4096 } },
        }, null);
    }
}

/// Checks each registered vision kernel launch against its shipped wrapper in `vision_kernels.zig`.
pub fn checkWrappers(a: std.mem.Allocator, items: []const kc.gates.Kernel) !void {
    var at: usize = 0;
    inline for (vis_dtypes) |dt| {
        const k = items[at];
        at += 1;
        const p: *VisQkvRope(dt) = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    }
    const has_nax = @import("../../mlx/device.zig").naxAvailable();
    inline for (nax_dtypes) |dt| {
        if (has_nax) {
            const k = items[at];
            const p: *VisAttnNax(dt) = @ptrCast(@alignCast(k.ctx));
            for (k.spec.served) |shape| {
                try std.testing.expect(try p.prepare(shape, .random, 7));
                try p.run();
                const ours = try a.dupe(u8, p.rig.outputBytes());
                defer a.free(ours);
                try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
            }
        }
        at += 1;
    }
    inline for (vis_dtypes) |dt| {
        inline for (.{ true, false }) |do_norm| {
            const k = items[at];
            at += 1;
            const p: *VisResNorm(dt, do_norm) = @ptrCast(@alignCast(k.ctx));
            for (k.spec.served) |shape| {
                try std.testing.expect(try p.prepare(shape, .random, 7));
                try p.run();
                const ours = try a.dupe(u8, p.rig.outputBytes());
                defer a.free(ours);
                try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
            }
        }
    }
    inline for (vis_dtypes) |dt| {
        inline for (.{ true, false }) |has_bias| {
            const k = items[at];
            at += 1;
            const p: *VisBiasGelu(dt, has_bias) = @ptrCast(@alignCast(k.ctx));
            for (k.spec.served) |shape| {
                try std.testing.expect(try p.prepare(shape, .random, 7));
                try p.run();
                const ours = try a.dupe(u8, p.rig.outputBytes());
                defer a.free(ours);
                try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
            }
        }
    }
}
