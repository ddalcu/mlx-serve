//! Inkling 2's verify reads in the kernel contract (`contract.zig` registers them): `relsplit_nax` (tensor unit) and
//! `relsplit_lanes` read 2 to 16 rows of up to four slots behind their caches in one pass.
//! Shape: (rows a slot) x slots x (slot 0's keys).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const source = @import("../../mlx/kernels.zig").source;
const device = @import("../../mlx/device.zig");
const attn = @import("../../sdk/blocks/attention/contract.zig");
const kernels = @import("kernels.zig");
const layers = @import("layers.zig");
const config = @import("config.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const KV = 1024;
const NONE: u32 = 0xffffffff;

fn ci(x: usize) c_int {
    return @intCast(x);
}

fn none(_: Shape) usize {
    return 0;
}

/// The picks tiled as `group.metal` tiles them: experts in order, each one's picks in pick order, up to four a tile;
/// past the last tile the expert is NONE and the count 0.
const Tiles = struct {
    texp: []u32,
    tcnt: []u32,
    tpick: []u32,
    used: usize = 0,

    fn of(a: std.mem.Allocator, picks: []const u32, experts: usize) !Tiles {
        const n = picks.len;
        var t = Tiles{ .texp = try a.alloc(u32, n), .tcnt = try a.alloc(u32, n), .tpick = try a.alloc(u32, 4 * n) };
        @memset(t.texp, NONE);
        @memset(t.tcnt, 0);
        @memset(t.tpick, 0);
        for (0..experts) |e| {
            var k: usize = 0;
            for (picks, 0..) |p, i| {
                if (p != e) continue;
                if (k % 4 == 0) {
                    t.texp[t.used] = @intCast(e);
                    t.used += 1;
                }
                t.tpick[4 * (t.used - 1) + k % 4] = @intCast(i);
                t.tcnt[t.used - 1] += 1;
                k += 1;
            }
        }
        return t;
    }

    /// TEXP, TCNT and TPICK as `rig`'s inputs.
    fn bind(t: Tiles, a: std.mem.Allocator, rig: *metal.Rig) ![3]mlx.mlx_array {
        var out: [3]mlx.mlx_array = undefined;
        for ([_][]const u32{ t.texp, t.tcnt, t.tpick }, &out) |v, *o| {
            const f = try a.alloc(f64, v.len);
            for (f, v) |*x, y| x.* = @floatFromInt(y);
            o.* = try rig.input(.uint32, &.{ci(v.len)}, f);
        }
        return out;
    }
};

// ── relsplit_nax, relsplit_lanes ──

/// KV heads, query heads a KV head, head width, relative extent and sliding window of every attention layer.
const HKV = 8;
const SG = 3;
const HQ = HKV * SG;
const SD = 128;
const R = 1024;
const WINDOW = 1024;
/// The log scaling's floor and alpha, runtime parameters (`sizes`, `params`): the proofs' floor, low enough that a few
/// hundred keys pass it.
const FLOOR = 256;
const ALPHA: f64 = @as(f32, 0.2);
/// The kernels' f32 log2(e), the score scale and their product.
const LOG2E: f64 = @as(f32, 1.4426950408889634);
const SCALE: f64 = @as(f32, 1.0 / @sqrt(@as(f32, SD)));
const QS: f64 = @as(f32, @floatCast(SCALE * LOG2E));
/// q, k and v enter `relsplit_lanes` as halves: exact for bf16 values but under half's normal range, where they move by
/// at most 2^-25.
const HALF_SUB = 0x1p-25;
const half_sub_row: [SD]f64 = @splat(HALF_SUB);
/// Keys a tile of the tensor-unit read and of the lanes read; slots a launch.
const NAX_BK = 32;
const LANES_BK = 16;
const LANES = kernels.nax_lanes;

const split_ins: []const [*:0]const u8 = &.{ "q", "k0", "k1", "k2", "k3", "v0", "v1", "v2", "v3", "kb", "vb", "rel", "sinks", "GATE", "TMETA", "sizes", "params" };
const split_outs: []const [*:0]const u8 = &.{ "PART", "ML", "OUT" };
const split_nax_launch = metal.Launch{ .name = "msv_inkling2_relsplit_nax", .inputs = split_ins, .outputs = split_outs, .source = source("plugins/inkling2/kernels/relsplit_nax.metal"), .header = source("kernels/common/mpp_tensor_ops.metal"), .contiguous = false, .tg_bytes = 0, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };
// The K and V tiles at their widest element: float, an f32 stream's.
const split_lanes_launch = metal.Launch{ .name = "msv_inkling2_relsplit_lanes", .inputs = split_ins, .outputs = split_outs, .source = source("plugins/inkling2/kernels/relsplit_lanes.metal"), .header = kernels.lanes_header, .contiguous = false, .tg_bytes = 4 * (SD * (LANES_BK + 8) + LANES_BK * (SD + 8)), .m1 = 1024, .m2 = 1024 };

/// What a split read serves: a prompt's rows inside their cache (one slot), a verify's rows behind each slot's cache
/// (their own keys a second source), or a tree verify's (a row sees its ancestors among them).
const Arm = enum { prompt, verify, tree };

fn outElem(dt: mlx.mlx_dtype) spec.Elem {
    return switch (dt) {
        .bfloat16 => .bf16,
        .float16 => .f16,
        else => .f32,
    };
}

/// What gate 3's bound allows an output of `e` whose value before its rounding is within `pre` of `want`.
fn allowOf(e: spec.Elem, want: f64, pre: f64) f64 {
    const half_ulp: f64 = switch (e) {
        .bf16 => 0x1p-8,
        .f16 => 0x1p-11,
        else => 0x1p-24,
    };
    const least: f64 = switch (e) {
        .bf16 => 0x1p-133,
        .f16 => 0x1p-24,
        else => 0x1p-149,
    };
    return pre + half_ulp * (@abs(want) + pre) + least;
}

/// A few rows of up to four slots read in one pass, each behind its own cache: (rows a slot) x slots x (slot 0's cached
/// keys, each later slot 7 fewer). One split: the rows normalised under their output gate (OUT); more: each split's
/// unnormalised sums (PART) and (max, sum) (ML).
fn Split(comptime nax: bool) type {
    return struct {
        rig: metal.Rig,
        launch: *const metal.Launch = if (nax) &split_nax_launch else &split_lanes_launch,
        arena: std.heap.ArenaAllocator,
        dt: mlx.mlx_dtype,
        window: usize,
        arm: Arm,
        /// A planted control: a grid one KV head short, its query heads never written.
        short: bool = false,
        shape: Shape = undefined,
        /// `[rows, HQ, SD]`; a slot's cache `[HKV, cached, SD]` twice; the rows' own keys and values `[HKV, rows, SD]`;
        /// the bias rows `[rows, HQ, R]`, sinks `[rows, HQ]` and gate `[rows, HQ, SD]`; a row's depth and ancestors.
        q: []f64 = &.{},
        ks: [LANES][]f64 = undefined,
        vs: [LANES][]f64 = undefined,
        kb: []f64 = &.{},
        vb: []f64 = &.{},
        rel: []f64 = &.{},
        sinks: []f64 = &.{},
        gate: []f64 = &.{},
        depth: []usize = &.{},
        mask: []u32 = &.{},
        args: [17]mlx.mlx_array = undefined,
        msg: [256]u8 = undefined,

        const Self = @This();
        const nax_only = nax;
        /// A slot's cache, its keys with the rows' own, its first visible key and its position.
        const Lane = struct { cached: usize, keys: usize, first: usize, offset: usize };
        const Plan = struct { kps: usize, splits: usize };

        fn init(a: std.mem.Allocator, dt: mlx.mlx_dtype, window: usize, arm: Arm) Self {
            return .{ .rig = metal.Rig.init(a), .arena = .init(a), .dt = dt, .window = window, .arm = arm };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        fn laneOf(self: *const Self, shape: Shape, z: usize) Lane {
            const cached = shape.k - 7 * z;
            const keys = cached + if (self.arm == .prompt) 0 else shape.rows;
            const first = if (self.window > 0 and keys + 1 > shape.rows + self.window) keys + 1 - shape.rows - self.window else 0;
            return .{ .cached = cached, .keys = keys, .first = first, .offset = if (self.arm == .prompt) cached - shape.rows else cached };
        }

        fn plan(self: *const Self, shape: Shape) Plan {
            var most: usize = 0;
            for (0..shape.n) |z| {
                const l = self.laneOf(shape, z);
                most = @max(most, l.keys - l.first);
            }
            const kps: usize = @intCast(kernels.splitKps(ci(most), shape.n));
            return .{ .kps = kps, .splits = @max(1, (most + kps - 1) / kps) };
        }

        pub fn geometry(self: *const Self, shape: Shape) ?metal.Geometry {
            const s = shape.rows;
            const lanes = shape.n;
            const least: usize = if (self.arm == .prompt) kernels.split_from else 2;
            if (lanes == 0 or lanes > LANES or s < least or s > kernels.open_rows) return null;
            if (self.arm == .prompt and lanes != 1) return null;
            // Every slot's cache holds two keys at least, a prompt's its own rows.
            if (shape.k < 7 * (lanes - 1) + (if (self.arm == .prompt) s else 2)) return null;
            const p = self.plan(shape);
            const lt = ci(lanes * s);
            var g = metal.Geometry{ .grid = .{ ci(p.splits * 32 * SG), HKV - @as(c_int, @intFromBool(self.short)), ci(lanes) }, .tg = .{ 32 * SG, 1, 1 } };
            if (p.splits == 1) {
                g.out(&.{1}, .float32);
                g.out(&.{1}, .float32);
                g.out(&.{ 1, lt, HQ * SD }, self.dt);
                g.unread = 0b011;
            } else {
                g.out(&.{ ci(p.splits), lt * HQ, SD }, .float32);
                g.out(&.{ ci(p.splits), lt * HQ, 2 }, .float32);
                g.out(&.{1}, .float32);
                g.unread = 0b100;
            }
            g.dtype("T", self.dt);
            g.int("D", @as(c_int, SD));
            g.int("HQ", @as(c_int, HQ));
            g.int("HKV", @as(c_int, HKV));
            g.int("R", @as(c_int, R));
            g.int("WINDOW", self.window);
            g.int("GLOBAL", @intFromBool(self.window == 0));
            g.int("DIRECT", @intFromBool(p.splits == 1));
            g.int("TREE", @intFromBool(self.arm == .tree));
            if (!nax) g.int("NR", @as(c_int, if (s > 8) 2 else 1));
            return g;
        }

        pub fn prepare(self: *Self, shape: Shape, fill: spec.Fill, seed: u64) !bool {
            if (self.geometry(shape) == null) return false;
            if (nax and !device.naxAvailable()) return error.NaxOnly;
            self.shape = shape;
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            const s = shape.rows;
            const lanes = shape.n;
            const lt = lanes * s;
            const own = self.arm != .prompt;
            self.q = try host.uniform(a, rng, lt * HQ * SD, -2, 2, self.dt);
            for (0..lanes) |z| {
                const l = self.laneOf(shape, z);
                self.ks[z] = try host.uniform(a, rng, HKV * l.cached * SD, -2, 2, self.dt);
                self.vs[z] = try host.uniform(a, rng, HKV * l.cached * SD, -2, 2, self.dt);
            }
            self.kb = &.{};
            self.vb = &.{};
            if (own) {
                self.kb = try host.uniform(a, rng, HKV * lt * SD, -2, 2, self.dt);
                self.vb = try host.uniform(a, rng, HKV * lt * SD, -2, 2, self.dt);
            }
            // bf16 values for an f32 stream too: the wrapper's identity projection of them is then exact in TF32.
            self.rel = try host.uniform(a, rng, lt * HQ * R, -2, 2, if (self.dt == .float32) .bfloat16 else self.dt);
            self.sinks = try host.uniform(a, rng, lt * HQ, -3, 3, self.dt);
            self.gate = try host.uniform(a, rng, lt * HQ * SD, 0.02, 0.99, .float32);
            // A tree verify's rows: row 0 the root, each later one a child of an earlier one.
            self.depth = try a.alloc(usize, lt);
            self.mask = try a.alloc(u32, lt);
            for (0..lanes) |z| {
                for (0..s) |r| {
                    const row = z * s + r;
                    if (self.arm != .tree or r == 0) {
                        self.depth[row] = if (self.arm == .tree) 0 else r;
                        self.mask[row] = if (self.arm == .tree) 1 else 0xffffffff;
                        continue;
                    }
                    const up = z * s + rng.uintLessThan(usize, r);
                    self.depth[row] = self.depth[up] + 1;
                    self.mask[row] = self.mask[up] | (@as(u32, 1) << @intCast(r));
                }
            }
            if (fill == .nonfinite) {
                host.plant(self.q, lt * HQ, SD);
                host.plant(self.vs[0], HKV * self.laneOf(shape, 0).cached, SD);
            }
            const p = self.plan(shape);
            var sizes: [35]f64 = @splat(0);
            for (0..lanes) |z| {
                const l = self.laneOf(shape, z);
                for (sizes[z * 8 ..][0..7], [_]usize{ s, l.keys, l.first, l.cached, l.offset, z * s, z * s }) |*o, v| o.* = @floatFromInt(v);
            }
            sizes[32] = @floatFromInt(p.kps);
            sizes[33] = FLOOR;
            sizes[34] = @floatFromInt(lt);
            const meta = try a.alloc(f64, 2 * lt);
            for (0..lt) |row| {
                meta[2 * row] = @floatFromInt(self.depth[row]);
                meta[2 * row + 1] = @floatFromInt(self.mask[row]);
            }
            self.rig.reset();
            var kin: [LANES]mlx.mlx_array = undefined;
            var vin: [LANES]mlx.mlx_array = undefined;
            for (0..LANES) |z| {
                // The launch binds four slots' caches; past its slots, the first again (never read).
                if (z >= lanes) {
                    kin[z] = kin[0];
                    vin[z] = vin[0];
                    continue;
                }
                const l = self.laneOf(shape, z);
                kin[z] = try self.rig.input(self.dt, &.{ 1, HKV, ci(l.cached), SD }, self.ks[z]);
                vin[z] = try self.rig.input(self.dt, &.{ 1, HKV, ci(l.cached), SD }, self.vs[z]);
            }
            const rows = ci(lt);
            self.args = .{
                try self.rig.input(self.dt, &.{ 1, rows, HQ, SD }, self.q),
                kin[0],
                kin[1],
                kin[2],
                kin[3],
                vin[0],
                vin[1],
                vin[2],
                vin[3],
                if (own) try self.rig.input(self.dt, &.{ 1, HKV, rows, SD }, self.kb) else kin[0],
                if (own) try self.rig.input(self.dt, &.{ 1, HKV, rows, SD }, self.vb) else vin[0],
                try self.rig.input(self.dt, &.{ 1, rows, HQ, R }, self.rel),
                try self.rig.input(self.dt, &.{ 1, rows, HQ }, self.sinks),
                try self.rig.input(.float32, &.{rows * HQ * SD}, self.gate),
                if (self.arm == .tree) try self.rig.input(.int32, &.{ rows, 2 }, meta) else try self.rig.input(.int32, &.{ 1, 2 }, &.{ 0, 0 }),
                try self.rig.input(.int32, &.{35}, &sizes),
                try self.rig.input(.float32, &.{2}, &.{ SCALE, ALPHA }),
            };
            return true;
        }

        pub fn run(self: *Self) !void {
            try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
        }

        pub fn count(self: *const Self) usize {
            const p = self.plan(self.shape);
            const n = self.shape.n * self.shape.rows * HQ;
            return if (p.splits == 1) n * SD else p.splits * n * (SD + 2);
        }

        /// P rounds to the matrices' element before P V: bf16 on the tensor unit's bf16 read, else half (2^-25 at least
        /// under its normal range); an f32 stream's P does not round, its products do.
        fn soft(self: *const Self) attn.Soft {
            if (self.dt == .float32) return .{ .base2 = true, .p_unit = host.U };
            if (nax and self.dt == .bfloat16) return .{ .base2 = true, .p_unit = 0x1p-8 };
            return .{ .base2 = true, .p_unit = 0x1p-11, .p_abs = 0x1p-25 };
        }

        /// A global layer's log scaling of a row at `offset + 1 + depth` past the floor, 1 + alpha ln(t / floor), with a
        /// fast-math log's error (4 ulps, 2^-21 at least), the divide's, and the f32 product's and add's.
        fn tauOf(self: *const Self, offset: usize, depth: usize) host.Val {
            const t = offset + 1 + depth;
            if (self.window != 0 or t <= FLOOR) return .exact(1);
            const ln = @log(@as(f64, @floatFromInt(t)) / FLOOR);
            const v = 1 + ALPHA * ln;
            const e_log = @max(8 * host.U * ln, 0x1p-21) + 2.5 * 0x1p-23;
            return .{ .v = v, .e = ALPHA * e_log + host.U * (ALPHA * ln + v) };
        }

        /// Key `j` of slot `z` (lane `l`) for its row `r` and query head `h`: its score and value, or masked (weight 0,
        /// its value still in the tile's P V) at `limit` or past it, ahead of the row, past the window or off its branch.
        fn keyAt(self: *const Self, l: Lane, z: usize, r: usize, h: usize, j: usize, limit: usize, tau: host.Val, terms: []f64, sim: attn.Sim) attn.Key {
            const s = self.shape.rows;
            const lt = self.shape.n * s;
            const row = z * s + r;
            const kv = h / SG;
            const own = j >= l.cached;
            const at = if (own) (kv * lt + z * s + j - l.cached) * SD else (kv * l.cached + j) * SD;
            const krow = if (own) self.kb[at..][0..SD] else self.ks[z][at..][0..SD];
            const vrow = if (own) self.vb[at..][0..SD] else self.vs[z][at..][0..SD];
            const ve: ?[]const f64 = if (!nax and self.dt == .bfloat16) &half_sub_row else null;
            const d: isize = @intCast(self.depth[row]);
            // A tree's own keys sit at their depth; every other key at its place behind the row's.
            const dist: isize = if (self.arm == .tree and own)
                d - @as(isize, @intCast(self.depth[z * s + j - l.cached]))
            else
                @as(isize, @intCast(l.keys - s)) + d - @as(isize, @intCast(j));
            const branch = self.arm != .tree or !own or ((self.mask[row] >> @intCast(j - l.cached)) & 1) != 0;
            const span: isize = @intCast(self.window);
            if (j >= limit or !branch or dist < 0 or (span > 0 and dist >= span)) return .{ .s = -std.math.inf(f64), .on = false, .v = vrow, .ve = ve };
            const q = self.q[(row * HQ + h) * SD ..][0..SD];
            // An f32 stream's products round too.
            const depth_qk: usize = if (self.dt == .float32) SD + 1 else SD;
            const dq = attn.dot(q, krow, QS, depth_qk, 1, sim, terms);
            var under: f64 = 0;
            if (ve != null) for (q, krow) |x, y| {
                under += @abs(x) + @abs(y);
            };
            const du: usize = @intCast(dist);
            const b: f64 = if (du < R) @as(f32, @floatCast(self.rel[(row * HQ + h) * R + du] * LOG2E)) else 0;
            const sv = dq.v + b;
            const e1 = dq.e + QS * HALF_SUB * under + host.U * (@abs(b) + @abs(sv));
            const sc = sv * tau.v;
            return .{ .s = sc, .e = e1 * tau.v + @abs(sv) * tau.e + host.U * @abs(sc), .v = vrow, .ve = ve };
        }

        /// Float64's outputs (or with `sim`'s q k sums) and, in `pre`, how far the device's may be before their own
        /// rounding: each split's online softmax over the kernel's tiles (the tensor unit's 32-key tiles of the cache
        /// and then of the rows' own keys, each read to its source's end; the lanes' 16-key tiles to the split's end).
        fn emit(self: *const Self, want: []f64, pre: ?[]f64, sim: attn.Sim) !void {
            const a = self.rig.a;
            const shape = self.shape;
            const s = shape.rows;
            const lt = shape.n * s;
            const p = self.plan(shape);
            const bk: usize = if (nax) NAX_BK else LANES_BK;
            const keys = try a.alloc(attn.Key, p.kps + 2 * bk);
            defer a.free(keys);
            const ends = try a.alloc(usize, p.kps / bk + 4);
            defer a.free(ends);
            const buf = try a.alloc(f64, 3 * SD);
            defer a.free(buf);
            const terms = buf[0..SD];
            const o = buf[SD .. 2 * SD];
            const eo = buf[2 * SD ..];
            const sf = self.soft();
            const parts = p.splits * lt * HQ;
            for (0..shape.n) |z| {
                const l = self.laneOf(shape, z);
                for (0..s) |r| {
                    const row = z * s + r;
                    const tau = self.tauOf(l.offset, self.depth[row]);
                    for (0..HQ) |h| for (0..p.splits) |sp| {
                        const kbeg = l.first + sp * p.kps;
                        const kend = @min(l.keys, kbeg + p.kps);
                        var n: usize = 0;
                        var nb: usize = 0;
                        var k0 = kbeg;
                        if (nax) {
                            const c_end = @min(kend, l.cached);
                            while (k0 < c_end) : (k0 += NAX_BK) {
                                for (k0..@min(k0 + NAX_BK, l.cached)) |j| {
                                    keys[n] = self.keyAt(l, z, r, h, j, c_end, tau, terms, sim);
                                    n += 1;
                                }
                                ends[nb] = n;
                                nb += 1;
                            }
                            k0 = @max(kbeg, l.cached);
                            while (k0 < kend) : (k0 += NAX_BK) {
                                for (k0..@min(k0 + NAX_BK, l.keys)) |j| {
                                    keys[n] = self.keyAt(l, z, r, h, j, kend, tau, terms, sim);
                                    n += 1;
                                }
                                ends[nb] = n;
                                nb += 1;
                            }
                        } else while (k0 < kend) : (k0 += LANES_BK) {
                            for (k0..@min(k0 + LANES_BK, kend)) |j| {
                                keys[n] = self.keyAt(l, z, r, h, j, kend, tau, terms, sim);
                                n += 1;
                            }
                            ends[nb] = n;
                            nb += 1;
                        }
                        const start: attn.Init = if (sp == 0) .{ .m = @as(f32, @floatCast(self.sinks[row * HQ + h] * LOG2E)), .l = 1 } else .{ .m = @as(f32, -1e30) };
                        const st = attn.online(sf, start, keys[0..n], ends[0..nb], o, eo);
                        if (p.splits == 1) {
                            attn.normalize(st, o, eo);
                            for (o, eo, 0..) |v, e, c| {
                                // Normalised in the input type, then times the gate in f32.
                                const mid = host.round(self.dt, .{ .v = v, .e = e });
                                const i = (row * HQ + h) * SD + c;
                                const gt = self.gate[i];
                                want[i] = mid.v * gt;
                                if (pre) |pe| pe[i] = mid.e * gt + host.U * @abs(want[i]);
                            }
                            continue;
                        }
                        const at = (sp * lt + row) * HQ + h;
                        @memcpy(want[at * SD ..][0..SD], o);
                        want[parts * SD + 2 * at] = st.m;
                        want[parts * SD + 2 * at + 1] = st.l;
                        if (pre) |pe| {
                            @memcpy(pe[at * SD ..][0..SD], eo);
                            pe[parts * SD + 2 * at] = st.em;
                            pe[parts * SD + 2 * at + 1] = st.el;
                        }
                    };
                }
            }
        }

        pub fn reference(self: *const Self, want: []f64, mag: []f64) !void {
            try self.emit(want, mag, null);
            for (mag) |*m| m.* = host.allowance(m.*);
        }

        pub fn simulate(self: *const Self, acc: spec.Accum, order: spec.Order, out: []f64) !void {
            try self.emit(out, null, .{ .acc = acc, .order = order });
        }

        /// The rule: within the allowance of `emit`, in the float64 output's class (a planted NaN or infinity reaches
        /// every row whose key tiles hold it, a masked key's value included).
        pub fn check(self: *Self, at: spec.Gate) !?[]const u8 {
            const a = self.rig.a;
            const n = self.count();
            const buf = try a.alloc(f64, 3 * n);
            defer a.free(buf);
            const want = buf[0..n];
            const allow = buf[n .. 2 * n];
            const got = buf[2 * n ..];
            try self.emit(want, allow, null);
            const e: spec.Elem = if (self.plan(self.shape).splits == 1) outElem(self.dt) else .f32;
            const bytes = self.rig.outputBytes();
            for (got, allow, want, 0..) |*g, *l, w, i| {
                g.* = spec.read(e, bytes, i);
                l.* = allowOf(e, w, l.*);
            }
            return host.within(&self.msg, at, got, want, allow);
        }

        /// The shipped read on the same inputs (its bias rows through an identity projection, exact), its splits joined
        /// by `relcomb` as the read joins them.
        fn parity(self: *Self) !void {
            const a = self.rig.a;
            var g = ops.Graph.init(a, self.rig.stream());
            defer g.deinit();
            const shape = self.shape;
            const lanes = shape.n;
            const x = self.args;
            const eye = try a.alloc(f32, R * R);
            defer a.free(eye);
            @memset(eye, 0);
            for (0..R) |i| eye[i * R + i] = 1;
            var L: layers.Layer = undefined;
            L.global = self.window == 0;
            L.rel_proj = try g.fromSlice(f32, eye, &.{ R, R });
            var cfg: config.Config = undefined;
            cfg.num_attention_heads = HQ;
            cfg.num_key_value_heads = HKV;
            cfg.head_dim = SD;
            cfg.sliding_window = WINDOW;
            cfg.log_n_floor = FLOOR;
            cfg.log_alpha = @floatCast(ALPHA);
            var offsets: [LANES]c_int = undefined;
            for (offsets[0..lanes], 0..) |*o, z| o.* = ci(self.laneOf(shape, z).offset);
            const own: ?[2]mlx.mlx_array = if (self.arm == .prompt) null else .{ x[9], x[10] };
            const meta: ?mlx.mlx_array = if (self.arm == .tree) x[14] else null;
            const s = ci(shape.rows);
            const read = if (nax)
                try kernels.splitReadNaxLanesTree(&g, &L, &cfg, x[0], x[11], x[12], x[1 .. 1 + lanes], x[5 .. 5 + lanes], own, offsets[0..lanes], x[13], s, meta)
            else
                try kernels.splitReadLanesM4(&g, &L, &cfg, x[0], x[11], x[12], x[1 .. 1 + lanes], x[5 .. 5 + lanes], own, offsets[0..lanes], x[13], s, meta);
            const theirs = read orelse return error.TestUnexpectedResult;
            const ours = if (self.plan(shape).splits == 1) self.rig.output(2) else try kernels.splitJoin(&g, .{ self.rig.output(0), self.rig.output(1) }, x[13], ci(lanes * shape.rows), self.dt);
            const want = try a.dupe(u8, try self.rig.bytesOf(&.{theirs}));
            defer a.free(want);
            try std.testing.expectEqualSlices(u8, want, try self.rig.bytesOf(&.{ours}));
        }
    };
}

const SplitCase = struct { dt: mlx.mlx_dtype, window: usize, arm: Arm, served: []const Shape };

// A verify of 2-4 slots behind a sliding window's keys, one slot past the window's start, a one-split read of four
// slots and of 16 rows; a tree verify behind a global layer's keys (log-scaled past the floor, its oldest tiles
// unmasked); a prompt's rows on each layer kind and in both stream dtypes.
const verify_sliding = SplitCase{ .dt = .bfloat16, .window = WINDOW, .arm = .verify, .served = &.{ .{ .rows = 4, .n = 2, .k = 300 }, .{ .rows = 9, .n = 1, .k = 1100 }, .{ .rows = 2, .n = 4, .k = 40 }, .{ .rows = 16, .n = 1, .k = 40 } } };
const tree_global = SplitCase{ .dt = .bfloat16, .window = 0, .arm = .tree, .served = &.{ .{ .rows = 8, .n = 2, .k = 300 }, .{ .rows = 4, .n = 1, .k = 1500 }, .{ .rows = 16, .n = 1, .k = 600 }, .{ .rows = 5, .n = 4, .k = 30 } } };
const nax_cases = [_]SplitCase{
    verify_sliding,
    tree_global,
    .{ .dt = .bfloat16, .window = WINDOW, .arm = .prompt, .served = &.{ .{ .rows = 4, .n = 1, .k = 40 }, .{ .rows = 8, .n = 1, .k = 1100 }, .{ .rows = 16, .n = 1, .k = 500 } } },
    .{ .dt = .float16, .window = 0, .arm = .prompt, .served = &.{ .{ .rows = 4, .n = 1, .k = 50 }, .{ .rows = 12, .n = 1, .k = 1300 }, .{ .rows = 16, .n = 1, .k = 64 } } },
};
// The lanes read serves verifies only, and an f32 stream's tree verify.
const lanes_cases = [_]SplitCase{
    verify_sliding,
    tree_global,
    .{ .dt = .float16, .window = 0, .arm = .verify, .served = &.{ .{ .rows = 4, .n = 2, .k = 300 }, .{ .rows = 12, .n = 1, .k = 1300 }, .{ .rows = 2, .n = 4, .k = 40 } } },
    .{ .dt = .float32, .window = 0, .arm = .tree, .served = &.{ .{ .rows = 4, .n = 2, .k = 300 }, .{ .rows = 8, .n = 1, .k = 40 } } },
};

fn splitName(comptime nax: bool, comptime c: SplitCase) []const u8 {
    const what = if (nax) "relsplit_nax" else "relsplit_lanes";
    const dt = switch (c.dt) {
        .bfloat16 => "bf16",
        .float16 => "f16",
        else => "f32",
    };
    const layer = if (c.window > 0) "sliding 1024" else "global";
    const arm = switch (c.arm) {
        .prompt => "prompt rows",
        .verify => "verify",
        .tree => "tree verify",
    };
    return "inkling2 " ++ what ++ " " ++ dt ++ " (" ++ layer ++ ", " ++ arm ++ ")";
}

const split_sums: spec.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the softmax's exp leaves no sum exact: the stress inputs check the fp32 sums within the float64 bound" };

fn splitSpec(comptime nax: bool, comptime name: []const u8, comptime served: []const Shape) spec.Spec {
    return .{
        .name = name,
        // The tensor unit's sums run in an order no simulation holds.
        .accum = if (nax) .none else .f32,
        .depth = host.depthOne,
        .out = .f32,
        .nan = .{ .stated = "a planted NaN or infinity reaches every row whose key tiles hold it, a masked key's value at weight 0 included" },
        .nax = nax,
        .served = served,
        .sums = if (nax) .{} else split_sums,
    };
}

// ── registration ──

// Every layout a pack's experts may carry, affine at every width at g64 and 4-bit at g32 and g128; the release's
// layouts (NVFP4 with its per-expert scales, MXFP8) and 4-bit affine at the full expert width too.
pub fn register(set: *kc.Set) !void {
    const a = set.a;
    inline for (nax_cases) |c| try set.add(Split(true), .init(a, c.dt, c.window, c.arm), splitSpec(true, splitName(true, c), c.served), null);
    inline for (lanes_cases) |c| try set.add(Split(false), .init(a, c.dt, c.window, c.arm), splitSpec(false, splitName(false, c), c.served), null);

    // The planted controls, last (`checkWrappers` walks the kernels before them): each launch with one fault its gate
    // must catch.
    inline for (.{ true, false }) |nax| {
        var split = Split(nax).init(a, .bfloat16, WINDOW, .verify);
        split.short = true;
        const name = "control inkling2 " ++ (if (nax) "relsplit_nax" else "relsplit_lanes") ++ " a KV head short (its query heads never written)";
        try set.add(Split(nax), split, splitSpec(nax, name, &.{.{ .rows = 2, .n = 1, .k = 40 }}), .shapes);
    }
}

/// `T`'s declared launch against its shipped wrapper at every served shape (the tensor-unit reads where it serves them).
fn drift(comptime T: type, k: kc.gates.Kernel) !void {
    if (comptime @hasDecl(T, "nax_only")) {
        if (T.nax_only and !device.naxAvailable()) return;
    }
    const p: *T = @ptrCast(@alignCast(k.ctx));
    for (k.spec.served) |shape| {
        try std.testing.expect(try p.prepare(shape, .random, 7));
        try p.run();
        try p.parity();
    }
}

/// The drift guard of `register`'s kernels, from the first (`items` starts there): each declared launch's bytes equal
/// the shipped wrapper's.
pub fn checkWrappers(items: []const kc.gates.Kernel) !void {
    var at: usize = 0;
    inline for (nax_cases) |_| {
        try drift(Split(true), items[at]);
        at += 1;
    }
    inline for (lanes_cases) |_| {
        try drift(Split(false), items[at]);
        at += 1;
    }
}
