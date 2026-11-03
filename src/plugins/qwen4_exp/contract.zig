//! qwen4_exp's kernel contract, listed in `kernel_contract/registry.zig`; each kernel bit-equal to the op chain it
//! replaces. The QSA pooled-key upkeep (`msv_qsa_pool_rope`: f32 block mean, rms norm, partial RoPE in one launch;
//! rows = pooled blocks, n = the 128 key dims, k = the block ratio) and the n-gram PLE gather (`msv_ple_gather`: hash,
//! row gather, dequant to bf16; rows = ids, n = heads x dim, k = the table's bits), bit-equal to the host gather. The
//! other kernels: `contract_*.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const pooled = @import("qsa/pooled.zig");
const ple_gpu = @import("ple_gpu.zig");
const qwen4 = @import("qwen4_exp.zig");
const Trunk = @import("trunk.zig").Trunk;

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

const D = 128;
/// The trunk's partial rotary span (256 x 0.25), and rotate_half's partner distance.
const RD = 64;
const HALF = RD / 2;
const EPS: f32 = 1e-6;

const pool_rope_launch: metal.Launch = .{
    .name = "mlxserve_qsa_pool_rope",
    .inputs = &.{ "kb", "norm_w", "cosv", "sinv", "eps", "inv_r", "nblk" },
    .outputs = &.{"out"},
    .source = pooled.QSA_POOL_ROPE_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const PoolRope = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &pool_rope_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    kb: []f64 = &.{},
    w: []f64 = &.{},
    cos: []f64 = &.{},
    sin: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) PoolRope {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *PoolRope) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// The wrapper's launch: one simdgroup a pooled row, lane l owning dims 4l..4l+3.
    pub fn geometry(_: *const PoolRope, shape: Shape) ?metal.Geometry {
        if (shape.n != D or shape.k < 1 or shape.k > 8 or shape.rows < 1) return null;
        const nb: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ 32, nb, 1 }, .tg = .{ 32, 1, 1 } };
        g.out(&.{ 1, nb, D }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("R", shape.k);
        g.int("RD", RD);
        g.int("HALF", HALF);
        return g;
    }

    pub fn prepare(self: *PoolRope, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const nb = shape.rows;
        const r = shape.k;
        self.kb = try a.alloc(f64, nb * r * D);
        self.w = try a.alloc(f64, D);
        self.cos = try a.alloc(f64, nb * RD);
        self.sin = try a.alloc(f64, nb * RD);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const bf = struct {
            fn of(v: f64) f64 {
                return spec.roundTo(.bf16, v);
            }
        }.of;
        for (self.kb) |*v| v.* = bf((rng.float(f64) * 2 - 1) * std.math.pow(f64, 10, rng.float(f64) * 1.5 - 1));
        for (self.w) |*v| v.* = bf(0.25 + 1.5 * rng.float(f64));
        for (self.cos, self.sin) |*c, *s| {
            const t = rng.float(f64) * 2 * std.math.pi;
            c.* = bf(@cos(t));
            s.* = bf(@sin(t));
        }
        switch (fill) {
            .random => {},
            .nonfinite => {
                // NaN, +Inf and -Inf in rows 0..2; row 3 holds +Inf and -Inf in one dim's block (its mean is NaN).
                const at = [_][2]usize{ .{ 0, D / 3 }, .{ 1, D / 2 }, .{ 2, D - 1 }, .{ 3, 5 }, .{ 3, (r - 1) * D + 5 } };
                const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
                for (at, put) |p, v| if (p[0] < nb) {
                    self.kb[p[0] * r * D + p[1]] = v;
                };
            },
            .stress => {
                // A block's rows agree (an exact mean); |x| in 32..64 makes 128 squares overflow fp16 in any order,
                // stay exact in fp32. Unit weights and cos 1, sin 0 leave only the sum of squares to miss.
                for (0..nb) |b| for (0..D) |d| {
                    const mag: f64 = @floatFromInt(32 + rng.uintLessThan(u32, 33));
                    const v = if (rng.boolean()) mag else -mag;
                    for (0..r) |i| self.kb[(b * r + i) * D + d] = v;
                };
                @memset(self.w, 1);
                @memset(self.cos, 1);
                @memset(self.sin, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const nbi: c_int = @intCast(nb);
        self.args[0] = try rig.input(.bfloat16, &.{ 1, nbi, @intCast(r), D }, self.kb);
        self.args[1] = try rig.input(.bfloat16, &.{D}, self.w);
        self.args[2] = try rig.input(.bfloat16, &.{ nbi, RD }, self.cos);
        self.args[3] = try rig.input(.bfloat16, &.{ nbi, RD }, self.sin);
        self.args[4] = try rig.scalar(EPS);
        self.args[5] = try rig.scalar(@floatCast(1.0 / @as(f64, @floatFromInt(r))));
        const nblk = mlx.mlx_array_new_int(nbi);
        try rig.ins.append(rig.a, .{ .parent = nblk, .view = mlx.mlx_array_new(), .len = 1 });
        self.args[6] = nblk;
        return true;
    }

    pub fn run(self: *PoolRope) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const PoolRope) usize {
        return self.shape.rows * D;
    }

    /// Every output: the block mean rounded to bf16 (it feeds the norm), the norm and rope unrounded (`mids`);
    /// with `acc`, the block sum and the sum of squares run in it.
    fn outputsIn(self: *const PoolRope, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) void {
        const r = self.shape.k;
        var terms: [D]f64 = undefined;
        for (0..self.shape.rows) |b| {
            var x: [D]f64 = undefined;
            for (&x, 0..) |*xd, d| {
                for (terms[0..r], 0..) |*t, i| t.* = self.kb[(b * r + i) * D + d];
                const sum = if (acc) |ac| spec.sumIn(ac, order, terms[0..r]) else sumF64(terms[0..r]);
                xd.* = spec.roundTo(.bf16, sum / @as(f64, @floatFromInt(r)));
            }
            for (&terms, x) |*t, v| t.* = v * v;
            const ss = if (acc) |ac| spec.sumIn(ac, order, &terms) else sumF64(&terms);
            const inv = 1 / @sqrt(ss / D + EPS);
            var n: [D]f64 = undefined;
            for (&n, x, self.w) |*o, v, w| o.* = w * v * inv;
            const out = want[b * D ..][0..D];
            for (out, 0..) |*o, d| {
                if (d >= RD) {
                    o.* = n[d];
                    continue;
                }
                const rot = if (d < HALF) -n[d + HALF] else n[d - HALF];
                o.* = n[d] * self.cos[b * RD + d] + rot * self.sin[b * RD + d];
            }
        }
        if (mag) |m| for (m, want) |*o, w| {
            o.* = @abs(w) / 2;
        };
    }

    pub fn reference(self: *const PoolRope, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const PoolRope, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.outputsIn(out, null, acc, order);
    }

    /// The composed chain on the same inputs: `qsaPoolNorm` then `qsaPooledRopeComposed`.
    pub fn replaced(self: *PoolRope) ![]const u8 {
        var xfm: Trunk = undefined;
        xfm.s = self.rig.stream();
        xfm.allocator = self.rig.a;
        xfm.config = .{};
        xfm.config.rms_norm_eps = EPS;
        const pn = try xfm.qsaPoolNorm(self.args[0], self.args[1]);
        defer _ = mlx.mlx_array_free(pn);
        const out = try xfm.qsaPooledRopeComposed(pn, RD, self.args[2], self.args[3]);
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *PoolRope) ![]const u8 {
        const out = (try pooled.qsaPoolNormRopeFused(self.rig.stream(), self.args[0], self.args[1], EPS, self.args[2], self.args[3], RD)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

const ple_launch: metal.Launch = .{
    .name = "msv_ple_gather",
    .inputs = &.{ "table", "ids", "prev", "params" },
    .outputs = &.{"out"},
    .source = ple_gpu.SOURCE,
    .header = ple_gpu.HEADER,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// The test tables: 1000 unigrams, trigrams, 4 heads an order (8 heads), 64 dims, eos 999.
const PLE_VOCAB = 1000;
const PLE_EOS = 999;
const PLE_DIM = 64;

const PleGather = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &ple_launch,
    hash: qwen4.NgramHash,
    /// One table per served width (4, 8 and 16 bits), written on first use.
    tables: [3]?ple_gpu.Fixture = .{ null, null, null },
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    ids: []u32 = &.{},
    prev: [2]u32 = undefined,
    host: std.ArrayList(u8) = .empty,
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) PleGather {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a), .hash = qwen4.NgramHash.init(PLE_VOCAB, 3, 4, 500, 1, 1234, 0, PLE_EOS) catch unreachable };
    }

    pub fn deinit(self: *PleGather) void {
        for (&self.tables) |*t| if (t.*) |*f| f.deinit();
        self.host.deinit(self.rig.a);
        self.rig.deinit();
        self.arena.deinit();
    }

    fn slot(bits: usize) ?usize {
        return switch (bits) {
            4 => 0,
            8 => 1,
            16 => 2,
            else => null,
        };
    }

    fn table(self: *PleGather) !*qwen4.NgramTable {
        const i = slot(self.shape.k).?;
        if (self.tables[i] == null) self.tables[i] = try ple_gpu.writeFixture(@intCast(self.shape.k), self.hash.total_rows, PLE_DIM, 32, 0x91e + i);
        return &self.tables[i].?.table;
    }

    fn tbl(t: *const qwen4.NgramTable) ple_gpu.Table {
        return .{ .arr = .{ .ctx = null }, .w_off = t.w_off, .s_off = t.s_off, .b_off = t.b_off };
    }

    /// The wrapper's launch: one thread an output element, 256 a group.
    pub fn geometry(self: *const PleGather, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n != self.hash.n_heads * PLE_DIM or slot(shape.k) == null) return null;
        const total: c_int = @intCast(shape.rows * shape.n);
        var g = metal.Geometry{ .grid = .{ total, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ @intCast(shape.rows), @intCast(shape.n) }, .bfloat16);
        return g;
    }

    /// Random ids; `.nonfinite` (the table holds finite normal bf16 by construction) plants the hash's edges
    /// instead: eos in the history and at every third id, the first and last unigram.
    pub fn prepare(self: *PleGather, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const t = try self.table();
        self.ids = try self.arena.allocator().alloc(u32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.ids) |*v| v.* = rng.uintLessThan(u32, PLE_VOCAB);
        self.prev = .{ rng.uintLessThan(u32, PLE_VOCAB), rng.uintLessThan(u32, PLE_VOCAB) };
        if (fill == .nonfinite) {
            self.prev[1] = PLE_EOS;
            for (self.ids, 0..) |*v, i| v.* = switch (i % 3) {
                0 => PLE_EOS,
                1 => 0,
                else => PLE_VOCAB - 2,
            };
        }
        const rig = &self.rig;
        rig.reset();
        self.args[0] = try rig.inputRaw(.uint8, &.{@intCast(t.map.len)}, t.map);
        var ids_i: [8192]i32 = undefined;
        for (self.ids, ids_i[0..self.ids.len]) |v, *o| o.* = @intCast(v);
        self.args[1] = try rig.inputRaw(.int32, &.{@intCast(self.ids.len)}, std.mem.sliceAsBytes(ids_i[0..self.ids.len]));
        const prev_i = [2]i32{ @intCast(self.prev[0]), @intCast(self.prev[1]) };
        self.args[2] = try rig.inputRaw(.int32, &.{2}, std.mem.sliceAsBytes(&prev_i));
        const p = ple_gpu.params(tbl(t), &self.hash, t, self.ids.len);
        self.args[3] = try rig.inputRaw(.int64, &.{ple_gpu.P_LEN}, std.mem.sliceAsBytes(&p));
        return true;
    }

    pub fn run(self: *PleGather) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const PleGather) usize {
        return self.shape.rows * self.shape.n;
    }

    /// The host gather's own bits stand as the reference (`check`, `replaced`).
    pub fn reference(_: *const PleGather, _: []f64, _: []f64) !void {
        return error.HostGatherIsTheReference;
    }

    pub fn check(self: *PleGather, _: spec.Gate) !?[]const u8 {
        const want = try self.replaced();
        if (!std.mem.eql(u8, want, self.rig.outputBytes())) return "a row differs from the host gather's bits";
        return null;
    }

    /// `NgramHash.rowIds` + `NgramTable.gather` + `bf16Rne` on the host, over the same ids and history.
    pub fn replaced(self: *PleGather) ![]const u8 {
        const rows = try ple_gpu.cpuRows(&self.hash, try self.table(), &self.prev, self.ids);
        defer std.testing.allocator.free(rows);
        self.host.clearRetainingCapacity();
        try self.host.appendSlice(self.rig.a, std.mem.sliceAsBytes(rows));
        return self.host.items;
    }

    /// The shipped wrapper on a copy of the same table.
    fn wrapper(self: *PleGather) ![]const u8 {
        const t = try self.table();
        var w = tbl(t);
        w.arr = mlx.mlx_array_new_data(t.map.ptr, &[_]c_int{@intCast(t.map.len)}, 1, .uint8);
        defer w.release();
        var ids_i: [8192]i32 = undefined;
        for (self.ids, ids_i[0..self.ids.len]) |v, *o| o.* = @intCast(v);
        const ids = mlx.mlx_array_new_data(&ids_i, &[_]c_int{@intCast(self.ids.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(ids);
        const out = try ple_gpu.embed(self.rig.stream(), w, &self.hash, t, ids, &self.prev);
        defer _ = mlx.mlx_array_free(out);
        return self.rig.bytesOf(&.{out});
    }
};

fn none(_: Shape) usize {
    return 0;
}

fn sumF64(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |v| s += v;
    return s;
}

/// The longest chain: the block sum, then the 128 squares.
fn depth(shape: Shape) usize {
    return shape.k + D;
}

pub fn register(set: *kc.Set) !void {
    try set.add(PoolRope, .init(set.a), .{
        .name = "qsa_pool_rope bf16 (mean, rms norm, partial rope)",
        .accum = .f32,
        .depth = depth,
        // f32 after the sums: the mean's 1/R, the mean square, + eps, rsqrt (2), x * inv.
        .ops = 6,
        // bf16 before the output's own rounding: x * inv, the weight, and rope's two products.
        .mids = 4,
        .out = .bf16,
        .replaces = "astype f32 -> mean -> astype bf16 -> fast::rms_norm -> rope (multiply, multiply, add)",
        // Decode (one block), a verify step crossing blocks, a 2048-token prefill chunk; ratio 4 on Flash Next.
        .served = &.{ .{ .rows = 1, .n = D, .k = 4 }, .{ .rows = 4, .n = D, .k = 4 }, .{ .rows = 512, .n = D, .k = 4 } },
        .sums = .{ .f16 = 32, .bf16 = 0, .why = "positive squares: a bf16 sum stays near the output's rounding" },
    }, null);
    try set.add(PleGather, .init(set.a), .{
        .name = "ple_gather bf16 (n-gram hash, row gather, dequant)",
        .accum = .none,
        .depth = none,
        .out = .bf16,
        .replaces = "NgramHash.rowIds -> NgramTable.gather -> bf16Rne (the host gather)",
        .nan = .{ .stated = "tables hold finite normal bf16 by construction; eos ids cut every n-gram as rowIds does" },
        .tiles = .{ .{ .size = 256 }, .{}, .{} },
        // A decode token, a verify step, a 2048-id prefill chunk at 4 bits; one token at 8 and 16 bits.
        .served = &.{ .{ .rows = 1, .n = 512, .k = 4 }, .{ .rows = 4, .n = 512, .k = 4 }, .{ .rows = 2048, .n = 512, .k = 4 }, .{ .rows = 3, .n = 512, .k = 8 }, .{ .rows = 2, .n = 512, .k = 16 } },
    }, null);
    try @import("contract_affine4.zig").register(set);
    try @import("contract_qsa_gather.zig").register(set);
    try @import("contract_qsa_attn.zig").register(set);
}

// ── tests ──

test "kernel contract qwen4_exp: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    inline for (.{ PoolRope, PleGather }, 0..) |T, i| {
        const k = set.kernels.items[i];
        const p: *T = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    }
}

test {
    _ = @import("contract_affine4.zig");
    _ = @import("contract_qsa_gather.zig");
    _ = @import("contract_qsa_attn.zig");
}
