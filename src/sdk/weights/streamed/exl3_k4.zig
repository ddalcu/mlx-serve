//! A layer whose EXL3 rate the stream lane's texts do not decode (a K outside `xk.bank_ks`: the 3.5 bpw packs' K = 4,
//! for the whole layer or its down projection) runs the stock tier's math (`experts.EagerChain`: per projection
//! `t128(inner(t128(x * rin[slot]))) * rout[slot]`, the clamped SwiGLU between) with sushi's rate-generic trellis inner
//! product for all three of its projections: the indexed GEMV for decode rows, the expert-sorted GEMM for prompt rows.
//! Every other layer runs the quant (`QuantMath`) unchanged. Sushi's inner products read f16 rows (exllamav3's).
const std = @import("std");
const sdk = @import("sdk");
const env = @import("mlx_host").env;
const mlx = sdk.mlx;
const sushi = @import("sushi_exl3");
const ops = @import("ops.zig");
const xp = @import("experts.zig");
const xk = @import("../../quant/exl3/stream/kernels.zig");
const quant = @import("sdk_ext.zig").quant;
const G = ops.MlxOps;
const T = mlx.mlx_array;
const Proj = xp.ProjOf(T);
const Bank = xp.BankArraysOf(T);
const n_banks = std.meta.fieldNames(xp.BankKind).len;

/// The lane's texts decode this code array's rate (16 K int16 a tile, K in `xk.bank_ks`).
pub fn laneDecodes(g: *G, code: T) bool {
    const last = g.shapeOf(code).d[3];
    if (@rem(last, 16) != 0) return false;
    return std.mem.indexOfScalar(u32, &xk.bank_ks, @intCast(@divExact(last, 16))) != null;
}

fn laneLayer(g: *G, b: Bank) bool {
    return laneDecodes(g, b.gate.code) and laneDecodes(g, b.up.code) and laneDecodes(g, b.down.code);
}

/// `128 ** -0.5` as the Python float rounds to the f32 MLX takes (`experts.t128_scale`).
const t128_scale: f32 = @bitCast(@as(u32, 0x3db504f3));

/// `tcq_runtime._t128`: the normalised 128-block Walsh-Hadamard over the last axis, in f32.
fn t128(g: *G, v: T) !T {
    const sh = g.shapeOf(v);
    var blocks = sh;
    blocks.d[sh.n - 1] = @divExact(sh.d[sh.n - 1], 128);
    blocks.d[sh.n] = 128;
    blocks.n += 1;
    const h = try g.hadamard(try g.reshape(try g.astype(v, .float32), blocks.slice()), t128_scale);
    return g.reshape(h, sh.slice());
}

/// The arm reads sushi's mul1 kernels: another codebook installed process-wide is refused by name.
fn requireMul1() !void {
    const d = sushi.kernels.activeDecode();
    if (!std.meta.eql(d, sushi.format.Decode.mul1)) return error.SushiDecodeNotMul1;
}

/// Which sushi inner product a call takes: decode rows by slot, or prompt rows sorted by slot.
const Inner = enum { indexed, sorted };

/// Test seam: the inner products (GEMV / GEMM dispatches) this arm has built.
var inner_dispatches: usize = 0;

/// One projection: rows `x` [rows, in] at slot rows `ids` [rows] (u32, ascending for `.sorted`) -> f32 [rows, out].
fn project(g: *G, x: T, ids: T, p: Proj, inner: Inner) !T {
    try requireMul1();
    if (@import("builtin").is_test) inner_dispatches += 1;
    const xh = try rotateIn(g, x, try g.take(p.rin, ids, 0));
    const z = try g.adopt(switch (inner) {
        .indexed => try sushi.kernels.indexedGemvCoopF16(g.s, xh, p.code, ids),
        .sorted => try sushi.kernels.innerGemmSorted(g.s, xh, p.code, ids),
    });
    return rotateOut(g, z, try g.take(p.rout, ids, 0));
}

/// The banked route's projection: every row through its own bank's arrays (`ids` packed bank << 24 | slot row), one
/// inner product over all the banks.
fn projectBanked(g: *G, x: T, ids: T, at: *const BankedRows, ps: [n_banks]Proj) !T {
    try requireMul1();
    if (@import("builtin").is_test) inner_dispatches += 1;
    var codes: [n_banks]T = undefined;
    for (&codes, ps) |*c, p| c.* = p.code;
    const xh = try rotateIn(g, x, try at.pick(g, ps, "rin"));
    const z = try g.adopt(try sushi.kernels.indexedGemvCoopBankedF16(g.s, xh, codes, ids));
    return rotateOut(g, z, try at.pick(g, ps, "rout"));
}

/// `t128(x * rin)` in f32, rounded to the f16 rows sushi's inner products read.
fn rotateIn(g: *G, x: T, rin_rows: T) !T {
    const rin = try g.astype(rin_rows, .float32);
    return g.astype(try t128(g, try g.mul(try g.astype(x, .float32), rin)), .float16);
}

/// `t128(z) * rout` in f32.
fn rotateOut(g: *G, z: T, rout_rows: T) !T {
    return g.mul(try t128(g, z), try g.astype(rout_rows, .float32));
}

/// `_clamped_swiglu` of gate and up (f32 [rows, inter]).
fn swiglu(g: *G, gg: T, uu: T, limit: f64) !T {
    const uc = try g.clip(uu, try g.scalar(-limit, .float32), try g.scalar(limit, .float32));
    const gc = try g.minimum(gg, try g.scalar(limit, .float32));
    return g.mul(try g.silu(gc), uc);
}

/// `QuantMath(G, Q)` with the arm for layers the lane's texts do not decode.
pub fn MixedKMath(comptime Q: type) type {
    const Lane = xp.QuantMath(G, Q);
    return struct {
        const Self = @This();
        pub const max_decode_rows: u32 = Lane.max_decode_rows;
        pub const has_banked = Lane.has_banked;
        pub const has_parts = Lane.has_parts;
        lane: Lane,
        swiglu_limit: f64,

        pub fn init(q: *Q, c: anytype) Self {
            return .{ .lane = Lane.init(q, c), .swiglu_limit = c.swiglu_limit };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: Proj, up: Proj) !T {
            if (laneDecodes(g, gate.code) and laneDecodes(g, up.code)) return self.lane.gateUp(g, x, ids, gate, up);
            return swiglu(g, try project(g, x, ids, gate, .indexed), try project(g, x, ids, up, .indexed), self.swiglu_limit);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: Proj) !T {
            if (laneDecodes(g, d.code)) return self.lane.down(g, h, ids, d);
            return project(g, h, ids, d, .indexed);
        }

        /// ids packed (bank << 24 | slot row): each row through its own bank's arrays, one inner product a projection.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [n_banks]Bank) !T {
            if (laneLayer(g, banks[0])) return self.lane.gateUpBanked(g, x, ids, banks);
            return bankedGateUp(g, x, ids, banks, self.swiglu_limit);
        }

        pub fn downBanked(self: *const Self, g: *G, h: T, ids: T, banks: *const [n_banks]Bank) !T {
            if (laneLayer(g, banks[0])) return self.lane.downBanked(g, h, ids, banks);
            return bankedDown(g, h, ids, banks);
        }

        /// A KEPT f32 [rows, hidden] in routed-row order, as the lane's waves return it.
        pub fn prefill(self: *Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: Bank) !T {
            if (laneLayer(g, bank)) return self.lane.prefill(g, layer, x, rows, bank);
            const m = g.mark();
            defer g.resetTo(m);
            const sorted = try sortedJoined(g, x, rows, bank, self.swiglu_limit, prompt_slice_rows);
            defer g.gpa.free(sorted.order);
            defer g.release(sorted.y);
            const back = try invert(g, sorted.order);
            return g.keep(try g.take(sorted.y, back, 0));
        }

        pub fn finishPrefill(self: *Self, g: *G) !void {
            return self.lane.finishPrefill(g);
        }

        /// `prefill` unjoined: a KEPT output a slice in slot order, and per row its routed row (`pos`).
        pub fn prefillParts(self: *Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: Bank, alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            if (laneLayer(g, bank)) return self.lane.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
            const order = try sortedCall(g, x, rows, bank, self.swiglu_limit, prompt_slice_rows, alloc, outs);
            defer g.gpa.free(order);
            try pos.appendSlice(alloc, order);
        }
    };
}

/// Sorted routed rows one prompt-arm slice runs: each slice is evaluated and its intermediates freed before the next,
/// so a call holds one slice's `promptRowBytes` beside the slices' outputs, whatever the call's width.
pub const prompt_slice_rows: usize = 8192;

/// Bytes the prompt arm builds per routed row of hidden `h` and inter `i` (bf16 rows in), whatever the layer's K:
/// every array a slice makes, so a bound however MLX schedules their frees. Gate and up 20h + 20i each (rin f16 +
/// f32, x f32, x * rin, its Hadamard, the f16 rows; the GEMM's f16, its f32 and Hadamard, rout f16 + f32, the
/// product), SwiGLU 16i, down 16i + 20h (its f32 output among them), the taken rows 2h, two u32 indices, and each
/// projection's sorted-GEMM window table (two u32 a window, at most a window a row).
pub fn promptRowBytes(h: u64, i: u64) u64 {
    return 62 * h + 72 * i + 8 + 3 * 2 * 4;
}

/// A slice's arrays that do not grow with its rows (SwiGLU's limits, the transforms' scalars): far under this.
const prompt_slice_fixed: u64 = 4096;

/// The prompt arm's bound for a call of `call_rows` routed rows: one slice's arrays, and per routed row its kept f32
/// output, the driver's join of the call's outputs and its reorder (4h each).
pub fn promptWorkBytes(h: u64, i: u64, call_rows: u64) u64 {
    return @min(call_rows, prompt_slice_rows) * promptRowBytes(h, i) + prompt_slice_fixed + call_rows * 3 * 4 * h;
}

/// Whether the lane's texts decode a layer of K `k` (its widest projection's); else the layer runs this arm.
pub fn laneDecodesK(k: u32) bool {
    return std.mem.indexOfScalar(u32, &xk.bank_ks, k) != null;
}

/// The call's routed rows sorted by slot (stable), through the three projections `slice` sorted rows at a time:
/// each slice's KEPT f32 [rows, hidden] (slot order, evaluated) appended to `outs`. Returns `order` (the caller frees
/// it): `order[i]` = the routed row at sorted position i.
fn sortedCall(g: *G, x: T, rows: quant.PrefillRows, bank: Bank, limit: f64, slice: usize, alloc: std.mem.Allocator, outs: *std.ArrayList(T)) ![]u32 {
    const n = rows.slot.len;
    const order = try g.gpa.alloc(u32, n);
    errdefer g.gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.sort.pdq(u32, order, rows.slot, struct {
        fn lt(slot: []const u32, a: u32, b: u32) bool {
            return slot[a] < slot[b] or (slot[a] == slot[b] and a < b);
        }
    }.lt);
    var at: usize = 0;
    while (at < n) : (at += slice) {
        try outs.ensureUnusedCapacity(alloc, 1);
        outs.appendAssumeCapacity(try sliceCall(g, x, rows, order[at..@min(at + slice, n)], bank, limit));
    }
    return order;
}

/// Sorted positions `order` through the three projections: a KEPT f32 [rows, hidden], evaluated, its intermediates freed.
fn sliceCall(g: *G, x: T, rows: quant.PrefillRows, order: []const u32, bank: Bank, limit: f64) !T {
    const m = g.mark();
    defer g.resetTo(m);
    const n = order.len;
    const slots = try g.gpa.alloc(u32, n);
    defer g.gpa.free(slots);
    const acts = try g.gpa.alloc(u32, n);
    defer g.gpa.free(acts);
    for (order, slots, acts) |o, *s, *ar| {
        s.* = rows.slot[o];
        ar.* = if (rows.act_row) |a| a[o] else o;
    }
    const ids = try g.hostArray(std.mem.sliceAsBytes(slots), &.{@intCast(n)}, .uint32);
    const xs = try g.take(x, try g.hostArray(std.mem.sliceAsBytes(acts), &.{@intCast(n)}, .uint32), 0);
    const h = try swiglu(g, try project(g, xs, ids, bank.gate, .sorted), try project(g, xs, ids, bank.up, .sorted), limit);
    const y = g.keep(try project(g, h, ids, bank.down, .sorted));
    errdefer g.release(y);
    try g.evalAll(&.{y});
    return y;
}

const Sorted = struct { y: T, order: []const u32 };

/// `sortedCall` joined: one KEPT f32 [rows, hidden] in slot order (the caller releases it and frees `order`).
fn sortedJoined(g: *G, x: T, rows: quant.PrefillRows, bank: Bank, limit: f64, slice: usize) !Sorted {
    var parts: std.ArrayList(T) = .empty;
    defer {
        for (parts.items) |p| g.release(p);
        parts.deinit(g.gpa);
    }
    const order = try sortedCall(g, x, rows, bank, limit, slice, g.gpa, &parts);
    errdefer g.gpa.free(order);
    const y = if (parts.items.len == 1) g.keep(parts.items[0]) else g.keep(try g.concat(parts.items, 0));
    return .{ .y = y, .order = order };
}

/// The banked route's gate/up and down: each row through its own bank's arrays.
fn bankedGateUp(g: *G, x: T, ids: T, banks: *const [n_banks]Bank, limit: f64) !T {
    const at = try BankedRows.of(g, ids);
    return swiglu(g, try projectBanked(g, x, ids, &at, projOf(banks, "gate")), try projectBanked(g, x, ids, &at, projOf(banks, "up")), limit);
}

fn bankedDown(g: *G, h: T, ids: T, banks: *const [n_banks]Bank) !T {
    const at = try BankedRows.of(g, ids);
    return projectBanked(g, h, ids, &at, projOf(banks, "down"));
}

fn projOf(banks: *const [n_banks]Bank, comptime which: []const u8) [n_banks]Proj {
    var out: [n_banks]Proj = undefined;
    for (&out, banks) |*o, *b| o.* = @field(b, which);
    return out;
}

/// The rows of `ids` (bank << 24 | slot) per bank: which rows it holds, and every row's slot in it (0 for another
/// bank's row, so a take reads inside the bank). Ids name banks 0 .. n_banks - 1.
const BankedRows = struct {
    mask: [n_banks]T,
    slot: [n_banks]T,

    fn of(g: *G, ids: T) !BankedRows {
        const shift = try u32Const(g, 1 << 24);
        const which = try g.floorDiv(ids, shift);
        const slot = try g.sub(ids, try g.mul(which, shift));
        const zero = try u32Const(g, 0);
        var at: BankedRows = undefined;
        for (&at.mask, &at.slot, 0..) |*m, *sl, b| {
            m.* = try g.equal(which, try u32Const(g, @intCast(b)));
            sl.* = try g.where(m.*, slot, zero);
        }
        return at;
    }

    /// Each row's `rin` or `rout` row from its own bank.
    fn pick(self: *const BankedRows, g: *G, ps: [n_banks]Proj, comptime field: []const u8) !T {
        var out = try g.take(@field(ps[n_banks - 1], field), self.slot[n_banks - 1], 0);
        var b: usize = n_banks - 1;
        while (b > 0) {
            b -= 1;
            out = try g.where(try g.expandDims(self.mask[b], 1), try g.take(@field(ps[b], field), self.slot[b], 0), out);
        }
        return out;
    }
};

/// A 0-d u32 constant (`MlxOps.scalar` makes float and int32 ones).
fn u32Const(g: *G, v: u32) !T {
    return g.hostArray(std.mem.asBytes(&v), &.{}, .uint32);
}

/// The inverse permutation of `order`, as a u32 index array.
fn invert(g: *G, order: []const u32) !T {
    const back = try g.gpa.alloc(u32, order.len);
    defer g.gpa.free(back);
    for (order, 0..) |o, i| back[o] = @intCast(i);
    return g.hostArray(std.mem.sliceAsBytes(back), &.{@intCast(order.len)}, .uint32);
}

const testing = std.testing;

/// `t128` in f64 on the host: 128-blocks of `v` through the Sylvester Hadamard, scaled by 1/sqrt(128).
fn hostT128(v: []f64) void {
    var tmp: [128]f64 = undefined;
    var at: usize = 0;
    while (at < v.len) : (at += 128) {
        for (0..128) |r| {
            var acc: f64 = 0;
            for (0..128) |c| acc += (if (@popCount(r & c) % 2 == 0) @as(f64, 1) else -1) * v[at + c];
            tmp[r] = acc / @sqrt(128.0);
        }
        @memcpy(v[at..][0..128], &tmp);
    }
}

test "exl3 k4 arm: a K4 projection, decode and prompt rows, equals the float64 reference of t128(t128(x * rin) @ W_hat) * rout" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    const E = 4;
    const in_dim = 256;
    const out_dim = 256;
    const K = 4;
    var prng = std.Random.DefaultPrng.init(0x4b34);
    const rnd = prng.random();
    const code = try a.alloc(i16, E * (in_dim / 16) * (out_dim / 16) * 16 * K);
    defer a.free(code);
    for (code) |*c| c.* = @bitCast(rnd.int(u16));
    var rin: [E * in_dim]f16 = undefined;
    var rout: [E * out_dim]f16 = undefined;
    for (&rin) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    for (&rout) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    xk.mul1Table(table);
    // W_hat [in, out] per expert from this lane's host decode (bit for bit exllamav3's, `sushi_parity`).
    const w = try a.alloc(u16, E * in_dim * out_dim);
    defer a.free(w);
    const per = (in_dim / 16) * (out_dim / 16) * 16 * K;
    for (0..E) |e| xk.reconstruct(code[e * per ..][0..per], in_dim / 16, out_dim / 16, K, table, w[e * in_dim * out_dim ..][0 .. in_dim * out_dim], null);
    const p: Proj = .{
        .code = try g.hostArray(std.mem.sliceAsBytes(code), &.{ E, in_dim / 16, out_dim / 16, 16 * K }, .int16),
        .rout = try g.hostArray(std.mem.sliceAsBytes(&rout), &.{ E, out_dim }, .float16),
        .rin = try g.hostArray(std.mem.sliceAsBytes(&rin), &.{ E, in_dim }, .float16),
    };
    for ([_]Inner{ .indexed, .sorted }) |inner| {
        const rows: usize = if (inner == .indexed) 3 else 20;
        const ids = try a.alloc(u32, rows);
        defer a.free(ids);
        for (ids, 0..) |*id, r| id.* = if (inner == .indexed) @intCast((r * 3 + 2) % E) else @intCast(r * E / rows);
        const x = try a.alloc(f32, rows * in_dim);
        defer a.free(x);
        for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
        const y = try project(&g, try g.hostArray(std.mem.sliceAsBytes(x), &.{ @intCast(rows), in_dim }, .float32), try g.hostArray(std.mem.sliceAsBytes(ids), &.{@intCast(rows)}, .uint32), p, inner);
        const got = try a.alloc(f32, rows * out_dim);
        defer a.free(got);
        _ = try g.hostF32(y, got);
        var dot: f64 = 0;
        var nw: f64 = 0;
        var ng: f64 = 0;
        var worst: f64 = 0;
        var peak: f64 = 0;
        for (0..rows) |r| {
            const e = ids[r];
            var xh: [in_dim]f64 = undefined;
            for (&xh, 0..) |*v, i| v.* = @as(f64, x[r * in_dim + i]) * @as(f64, rin[e * in_dim + i]);
            hostT128(&xh);
            var z: [out_dim]f64 = @splat(0);
            for (0..in_dim) |i| for (0..out_dim) |o| {
                z[o] += xh[i] * @as(f64, @as(f16, @bitCast(w[(e * in_dim + i) * out_dim + o])));
            };
            hostT128(&z);
            for (0..out_dim) |o| {
                const want = z[o] * @as(f64, rout[e * out_dim + o]);
                const have: f64 = got[r * out_dim + o];
                try testing.expect(std.math.isFinite(have));
                dot += want * have;
                nw += want * want;
                ng += have * have;
                worst = @max(worst, @abs(want - have));
                peak = @max(peak, @abs(want));
            }
        }
        // The arm rounds t128(x * rin) and the inner product to f16 (exllamav3's rows): within f16 steps of the truth.
        try testing.expect(dot / @sqrt(nw * ng) > 0.99999);
        try testing.expect(worst <= 4e-3 * peak);
    }
}

/// A K4 bank of `e` experts (hidden and inter 256) with random codes and scales in [0.5, 1.5).
fn testBank(g: *G, a: std.mem.Allocator, rnd: std.Random, e: c_int) !Bank {
    return testBankK(g, a, rnd, e, .{ 4, 4, 4 });
}

/// `testBank` with each projection's K (gate, up, down).
fn testBankK(g: *G, a: std.mem.Allocator, rnd: std.Random, e: c_int, kp: [3]c_int) !Bank {
    return testBankDims(g, a, rnd, e, kp, 256, 256);
}

/// A bank of `e` experts, hidden `h` and inter `i`, each projection's K, random codes and scales in [0.5, 1.5).
fn testBankDims(g: *G, a: std.mem.Allocator, rnd: std.Random, e: c_int, kp: [3]c_int, h: c_int, i: c_int) !Bank {
    var projs: [3]Proj = undefined;
    const dims = [3][2]c_int{ .{ h, i }, .{ h, i }, .{ i, h } };
    for (&projs, kp, dims) |*p, k, d| {
        const code = try a.alloc(i16, @as(usize, @intCast(e * @divExact(d[0], 16) * @divExact(d[1], 16) * 16 * k)));
        defer a.free(code);
        for (code) |*c| c.* = @bitCast(rnd.int(u16));
        const rin = try a.alloc(f16, @intCast(e * d[0]));
        defer a.free(rin);
        for (rin) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
        const rout = try a.alloc(f16, @intCast(e * d[1]));
        defer a.free(rout);
        for (rout) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
        p.* = .{
            .code = try g.hostArray(std.mem.sliceAsBytes(code), &.{ e, @divExact(d[0], 16), @divExact(d[1], 16), 16 * k }, .int16),
            .rout = try g.hostArray(std.mem.sliceAsBytes(rout), &.{ e, d[1] }, .float16),
            .rin = try g.hostArray(std.mem.sliceAsBytes(rin), &.{ e, d[0] }, .float16),
        };
    }
    return .{ .gate = projs[0], .up = projs[1], .down = projs[2] };
}

/// One routed row through its own bank alone, decode rows: the per-row reference the batched routes must keep.
fn rowAlone(g: *G, x_row: T, slot: u32, b: Bank, limit: f64) !T {
    const one = try g.hostArray(std.mem.asBytes(&slot), &.{1}, .uint32);
    const h = try swiglu(g, try project(g, x_row, one, b.gate, .indexed), try project(g, x_row, one, b.up, .indexed), limit);
    return project(g, h, one, b.down, .indexed);
}

fn expectRowsNear(g: *G, a: std.mem.Allocator, got: T, want: T, rel: f64) !void {
    const n: usize = @intCast(mlx.mlx_array_size(want));
    const gv = try a.alloc(f32, n);
    defer a.free(gv);
    const wv = try a.alloc(f32, n);
    defer a.free(wv);
    _ = try g.hostF32(got, gv);
    _ = try g.hostF32(want, wv);
    var peak: f64 = 0;
    for (wv) |v| peak = @max(peak, @abs(@as(f64, v)));
    try testing.expect(peak > 0);
    for (gv, wv) |x, y| {
        try testing.expect(std.math.isFinite(x));
        try testing.expect(@abs(@as(f64, x) - y) <= rel * peak);
    }
}

test "exl3 k4 arm: the banked decode route serves each row from its own bank as that row alone would be served" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xba4c);
    const rnd = prng.random();
    const banks = [n_banks]Bank{ try testBank(&g, a, rnd, 5), try testBank(&g, a, rnd, 3), try testBank(&g, a, rnd, 4) };
    // (bank, slot) per row: every bank, a slot past the smaller banks' rows, repeats.
    const routed = [_][2]u32{ .{ 0, 4 }, .{ 2, 1 }, .{ 1, 2 }, .{ 0, 0 }, .{ 2, 3 }, .{ 0, 4 } };
    var packed_ids: [routed.len]u32 = undefined;
    for (routed, &packed_ids) |r, *p| p.* = r[0] << 24 | r[1];
    var xv: [routed.len * 256]f32 = undefined;
    for (&xv) |*v| v.* = rnd.float(f32) * 2 - 1;
    const x = try g.hostArray(std.mem.sliceAsBytes(&xv), &.{ routed.len, 256 }, .float32);
    const ids = try g.hostArray(std.mem.sliceAsBytes(&packed_ids), &.{routed.len}, .uint32);
    const limit: f64 = 7;
    const y = try bankedDown(&g, try bankedGateUp(&g, x, ids, &banks, limit), ids, &banks);
    var alone: [routed.len]T = undefined;
    for (routed, &alone, 0..) |r, *o, i| o.* = try rowAlone(&g, try g.reshape(try g.take(x, try g.scalar(@floatFromInt(i), .int32), 0), &.{ 1, 256 }), r[1], banks[r[0]], limit);
    try expectRowsNear(&g, a, y, try g.concat(&alone, 0), 1e-5);
}

/// Equal values (NaN where NaN), element for element.
fn expectRowsExact(g: *G, a: std.mem.Allocator, got: T, want: T) !void {
    const n: usize = @intCast(mlx.mlx_array_size(want));
    try testing.expectEqual(n, @as(usize, @intCast(mlx.mlx_array_size(got))));
    const gv = try a.alloc(f32, n);
    defer a.free(gv);
    const wv = try a.alloc(f32, n);
    defer a.free(wv);
    _ = try g.hostF32(got, gv);
    _ = try g.hostF32(want, wv);
    for (gv, wv) |x, y| try testing.expect(x == y or (std.math.isNan(x) and std.math.isNan(y)));
}

const BankedCase = struct { banks: [n_banks]Bank, routed: []const [2]u32 };

/// The banked decode route (gate/up then down) over `routed` (bank, slot) rows of random inputs, and the same rows
/// each alone through its own bank.
fn bankedAndAlone(g: *G, rnd: std.Random, c: BankedCase, limit: f64) ![2]T {
    var packed_ids: [8]u32 = undefined;
    for (c.routed, packed_ids[0..c.routed.len]) |r, *p| p.* = r[0] << 24 | r[1];
    var xv: [8 * 256]f32 = undefined;
    for (xv[0 .. c.routed.len * 256]) |*v| v.* = rnd.float(f32) * 2 - 1;
    const rows: c_int = @intCast(c.routed.len);
    const x = try g.hostArray(std.mem.sliceAsBytes(xv[0 .. c.routed.len * 256]), &.{ rows, 256 }, .float32);
    const ids = try g.hostArray(std.mem.sliceAsBytes(packed_ids[0..c.routed.len]), &.{rows}, .uint32);
    const y = try bankedDown(g, try bankedGateUp(g, x, ids, &c.banks, limit), ids, &c.banks);
    var alone: [8]T = undefined;
    for (c.routed, alone[0..c.routed.len], 0..) |r, *o, i| o.* = try rowAlone(g, try g.reshape(try g.take(x, try g.scalar(@floatFromInt(i), .int32), 0), &.{ 1, 256 }), r[1], c.banks[r[0]], limit);
    return .{ y, try g.concat(alone[0..c.routed.len], 0) };
}

test "exl3 k4 arm: the banked decode route equals each row alone exactly, at any mix of banks, K4 and 3/3/4 layers" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xb4e5);
    const rnd = prng.random();
    const limit: f64 = 7;
    for ([_][3]c_int{ .{ 4, 4, 4 }, .{ 3, 3, 4 } }) |kp| {
        const b0 = try testBankK(&g, a, rnd, 5, kp);
        const b1 = try testBankK(&g, a, rnd, 3, kp);
        const b2 = try testBankK(&g, a, rnd, 4, kp);
        const cases = [_]BankedCase{
            // Every bank, repeats, a slot past the smaller banks' rows.
            .{ .banks = .{ b0, b1, b2 }, .routed = &.{ .{ 0, 4 }, .{ 2, 1 }, .{ 1, 2 }, .{ 0, 0 }, .{ 2, 3 }, .{ 0, 4 } } },
            // One bank (a miss wave's transient rows).
            .{ .banks = .{ b0, b1, b2 }, .routed = &.{ .{ 2, 3 }, .{ 2, 0 }, .{ 2, 3 } } },
            // A bank no row names (DEVROUTE's hit wave never names the transient bank).
            .{ .banks = .{ b0, b1, b2 }, .routed = &.{ .{ 1, 2 }, .{ 0, 4 }, .{ 0, 1 }, .{ 1, 0 } } },
            // A bank without rows given another's arrays (`bankedArrays`' fill).
            .{ .banks = .{ b0, b1, b0 }, .routed = &.{ .{ 0, 3 }, .{ 1, 1 } } },
            // One row.
            .{ .banks = .{ b0, b1, b2 }, .routed = &.{.{ 1, 1 }} },
        };
        for (cases) |c| {
            const got = try bankedAndAlone(&g, rnd, c, limit);
            try expectRowsExact(&g, a, got[0], got[1]);
        }
        // Control: the fixture tells the banks apart (bank 0's slot 1 is not bank 2's slot 1).
        var xv: [256]f32 = undefined;
        for (&xv) |*v| v.* = rnd.float(f32) * 2 - 1;
        const x = try g.hostArray(std.mem.sliceAsBytes(&xv), &.{ 1, 256 }, .float32);
        try testing.expectError(error.TestUnexpectedResult, expectRowsExact(&g, a, try rowAlone(&g, x, 1, b0, limit), try rowAlone(&g, x, 1, b2, limit)));
    }
}

test "exl3 k4 arm: a banked decode call builds one inner product per projection, whatever its banks" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xd15c);
    const rnd = prng.random();
    const banks = [n_banks]Bank{ try testBank(&g, a, rnd, 5), try testBank(&g, a, rnd, 3), try testBank(&g, a, rnd, 4) };
    for ([_][]const [2]u32{ &.{ .{ 2, 3 }, .{ 2, 0 } }, &.{ .{ 0, 4 }, .{ 1, 2 }, .{ 2, 1 } } }) |routed| {
        inner_dispatches = 0;
        _ = try bankedAndAlone(&g, rnd, .{ .banks = banks, .routed = routed }, 7);
        // gate, up, down of the banked call, then each row alone's three.
        try testing.expectEqual(3 + 3 * routed.len, inner_dispatches);
    }
}

test "exl3 k4 arm: prompt rows sorted by slot come back in routed order, and the unjoined parts name each row" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x5047);
    const rnd = prng.random();
    const bank = try testBank(&g, a, rnd, 6);
    const slot = [_]u32{ 5, 1, 1, 3, 0, 5, 2, 4, 1, 0, 3, 5 };
    var act: [slot.len]u32 = undefined;
    for (&act, 0..) |*v, i| v.* = @intCast(i % 7);
    var xv: [7 * 256]f32 = undefined;
    for (&xv) |*v| v.* = rnd.float(f32) * 2 - 1;
    const x = try g.hostArray(std.mem.sliceAsBytes(&xv), &.{ 7, 256 }, .float32);
    const limit: f64 = 7;
    const sorted = try sortedJoined(&g, x, .{ .slot = &slot, .act_row = &act }, bank, limit, prompt_slice_rows);
    defer g.gpa.free(sorted.order);
    defer g.release(sorted.y);
    for (sorted.order[1..], sorted.order[0 .. slot.len - 1]) |b, prev| try testing.expect(slot[prev] < slot[b] or (slot[prev] == slot[b] and prev < b));
    const routed = try g.take(sorted.y, try invert(&g, sorted.order), 0);
    var alone: [slot.len]T = undefined;
    for (slot, act, &alone) |sl, ar, *o| o.* = try rowAlone(&g, try g.reshape(try g.take(x, try g.scalar(@floatFromInt(ar), .int32), 0), &.{ 1, 256 }), sl, bank, limit);
    // The sorted GEMM and the indexed GEMV are two kernels over the same f16 rows: within f16 steps of each other.
    try expectRowsNear(&g, a, routed, try g.concat(&alone, 0), 4e-3);
    // The unjoined parts: sorted position i is routed row order[i].
    for (sorted.order, 0..) |r, i| try expectRowsNear(&g, a, try g.take(sorted.y, try g.scalar(@floatFromInt(i), .int32), 0), try g.take(routed, try g.scalar(@floatFromInt(r), .int32), 0), 0);
}

test "exl3 k4 arm: prompt rows run in slices equal the whole call bit for bit, K4 and 3/3/4" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x511c);
    const rnd = prng.random();
    for ([_][3]c_int{ .{ 4, 4, 4 }, .{ 3, 3, 4 } }) |kp| {
        const bank = try testBankK(&g, a, rnd, 5, kp);
        // Runs longer than a 32-row window and than a slice, every slot; slices cut inside runs.
        var slot: [150]u32 = undefined;
        for (&slot, 0..) |*v, i| v.* = if (i < 70) 1 else rnd.uintLessThan(u32, 5);
        var act: [slot.len]u32 = undefined;
        for (&act, 0..) |*v, i| v.* = @intCast(i % 29);
        var xv: [29 * 256]f32 = undefined;
        for (&xv) |*v| v.* = rnd.float(f32) * 2 - 1;
        const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&xv), &.{ 29, 256 }, .float32), .bfloat16);
        const rows: quant.PrefillRows = .{ .slot = &slot, .act_row = &act };
        const whole = try sortedJoined(&g, x, rows, bank, 7, slot.len);
        defer g.gpa.free(whole.order);
        defer g.release(whole.y);
        for ([_]usize{ 1, 7, 32, 64 }) |slice| {
            const cut = try sortedJoined(&g, x, rows, bank, 7, slice);
            defer g.gpa.free(cut.order);
            defer g.release(cut.y);
            try testing.expectEqualSlices(u32, whole.order, cut.order);
            try expectRowsExact(&g, a, cut.y, whole.y);
        }
    }
}

test "exl3 k4 arm: a sliced prompt call stays within promptWorkBytes at real widths, K4 and 3/3/4; the whole call does not" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x9eac);
    const rnd = prng.random();
    const h = 5120;
    const i = 2304;
    const n = 4096;
    const slice = 1024;
    for ([_][3]c_int{ .{ 4, 4, 4 }, .{ 3, 3, 4 } }) |kp| {
        const bank = try testBankDims(&g, a, rnd, 4, kp, h, i);
        const slot = try a.alloc(u32, n);
        defer a.free(slot);
        const act = try a.alloc(u32, n);
        defer a.free(act);
        for (slot, act, 0..) |*sv, *av, r| {
            sv.* = rnd.uintLessThan(u32, 4);
            av.* = @intCast(r / 6);
        }
        const xv = try a.alloc(f32, (n / 6 + 1) * h);
        defer a.free(xv);
        for (xv) |*v| v.* = rnd.float(f32) * 2 - 1;
        const x = g.keep(try g.astype(try g.hostArray(std.mem.sliceAsBytes(xv), &.{ n / 6 + 1, h }, .float32), .bfloat16));
        defer g.release(x);
        try g.evalAll(&.{x});
        try mlx.check(mlx.mlx_array_eval(bank.gate.code));
        for ([_]T{ bank.gate.rin, bank.gate.rout, bank.up.code, bank.up.rin, bank.up.rout, bank.down.code, bank.down.rin, bank.down.rout }) |w| try mlx.check(mlx.mlx_array_eval(w));
        const rows: quant.PrefillRows = .{ .slot = slot, .act_row = act };
        // The unjoined path the source runs: each slice's output kept, the call's arrays released at its reset.
        var peaks: [2]usize = undefined;
        for ([_]usize{ slice, n }, &peaks) |width, *peak| {
            _ = mlx.mlx_synchronize(s);
            _ = mlx.mlx_clear_cache();
            var active: usize = 0;
            _ = mlx.mlx_get_active_memory(&active);
            _ = mlx.mlx_reset_peak_memory();
            var outs: std.ArrayList(T) = .empty;
            defer outs.deinit(a);
            const m = g.mark();
            const order = try sortedCall(&g, x, rows, bank, 7, width, a, &outs);
            g.gpa.free(order);
            _ = mlx.mlx_get_peak_memory(peak);
            peak.* -= active;
            g.resetTo(m);
            for (outs.items) |o| g.release(o);
        }
        const bound = slice * promptRowBytes(h, i) + prompt_slice_fixed + n * 4 * h;
        try testing.expect(peaks[0] <= bound);
        // Control: unsliced, the call holds every row's arrays (the per-row count is their sum), past the bound.
        try testing.expect(peaks[1] > bound);
        try testing.expect(peaks[1] <= n * promptRowBytes(h, i) + prompt_slice_fixed);
    }
}
