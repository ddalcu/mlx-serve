//! The MoE down half of a decode block's 1..6 rows in one launch: the routed and shared down projections, their combine
//! and the hc expand into the next stream. A row's `y` is its picked experts' down rows plus the shared expert's,
//! rounded to the stream's dtype once; the new stream is `post * y + comb^T x1` (`hcExpand`). Each distinct expert the
//! rows pick is read once. The sums run in f32 over the experts and the reduction axis together: not slot by slot, and
//! without the bf16 rounding the chain puts on each expert's row.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mxfp8_qmv = @import("../../../sdk/blocks/qmm/mxfp8_qmv.zig");
const params = @import("../../../sdk/quant/params.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const TOPK = 6;
const HC = 4;

/// Output rows a threadgroup owns, a thread each per 128 inputs (576 threads at the release's K). `probe` halves it
/// where the GPU grants fewer threads (M1/M2).
pub const rows_per_tg = 32;

/// Threadgroup bytes of an instantiation: the staged terms (148 floats a 128-value unit) and the groups' sums, both
/// double buffered, and the job tables.
fn tgBytes(K: usize, rows: usize) usize {
    const units = K / 128;
    return 4 * (2 * units * 148 + 2 * units * 2 + 4 * rows * TOPK);
}

/// The widest reduction axis whose threadgroup memory fits the 32 KiB every Apple GPU grants.
pub const max_k = blk: {
    var k: usize = 128;
    while (tgBytes(k + 128, decode.max_rows) <= 32 * 1024) k += 128;
    break :blk k;
};

pub const INS = [_][*:0]const u8{ "x", "w", "scales", "biases", "sw", "sscales", "inds", "res", "post", "comb" };

/// A thread owns one output row and one 128-value unit of the reduction axis (48 bytes of 3-bit codes, 128 of fp8). The
/// threadgroup stages each pair's activation row once in threadgroup memory (terms prescaled by their field's shift, so a
/// field is masked, never shifted), and walks the jobs: the row pairs ordered by expert, an expert's weights loading once.
pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_down.metal");
pub const HEADER = mxfp8_qmv.HEADER ++ kernelSource("plugins/deepseek_v41/kernels/dec_moe_down_header.metal");

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_moe_down", &INS, &.{"out"}, SOURCE, HEADER);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dims(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

/// x' `[R, hc, N]` = `hcExpand(x1, y, sp)` for y = the `idx` `[R, 6]` experts' down rows plus the shared expert's over
/// `act` `[R, 7, K]` (the 6 weighted routed slots, then the shared: what `moe_up` makes). `down` is the routed bank
/// (the one-expert view of `Experts.views`: the kernel addresses experts from its buffer pointer), `shared` the mxfp8
/// down linear, `rb` what `probe` returned. Null where the kernel does not serve the call: more than `max_rows` rows,
/// a down that is not 3-bit group-64 affine over bf16 scales, a shared that is not mxfp8, a stream that is not bf16,
/// K not a multiple of 128 or past `max_k`, or N not a multiple of `rb`.
pub fn moeDown(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, rb: usize, down: *const dsv41.Bank, shared: *const dsv41.Lin, act: A, idx: A, x1: A, sp: *const dsv41.Split) !?A {
    if (!mlx.streamIsGpu(sc.s)) return null;
    if (mlx.getShape(act).len != 3 or mlx.getShape(x1).len != 3 or mlx.getShape(idx).len != 2) return null;
    const R = dims(act, 0);
    const K = dims(act, 2);
    const N = dims(x1, 2);
    const nt = rb * (K / 128);
    if (R < 1 or R > decode.max_rows or dims(act, 1) != TOPK + 1 or dims(x1, 0) != R or dims(x1, 1) != HC or dims(idx, 0) != R or dims(idx, 1) != TOPK) return null;
    if (K == 0 or K % 128 != 0 or K > max_k or N % rb != 0 or sp.post.ctx == null or sp.comb.ctx == null) return null;
    if (down.qp.mode != .affine or down.qp.bits != 3 or down.qp.group_size != 64 or down.b.ctx == null) return null;
    if (shared.qp.mode != .mxfp8 or shared.qp.bits != 8 or shared.qp.group_size != 32) return null;
    const bfd: mlx.mlx_dtype = .bfloat16;
    if (mlx.mlx_array_dtype(act) != bfd or mlx.mlx_array_dtype(x1) != bfd or mlx.mlx_array_dtype(down.s) != bfd or mlx.mlx_array_dtype(down.b) != bfd) return null;
    if (mlx.mlx_array_dtype(down.w) != .uint32 or mlx.mlx_array_dtype(shared.w) != .uint32 or mlx.mlx_array_dtype(shared.s) != .uint8) return null;
    if (mlx.mlx_array_dtype(sp.post) != .float32 or mlx.mlx_array_dtype(sp.comb) != .float32 or mlx.mlx_array_size(sp.post) != R * HC or mlx.mlx_array_size(sp.comb) != R * HC * HC) return null;
    if (dims(down.w, 1) != N or dims(down.w, 2) != K * 3 / 32 or dims(down.s, 2) != K / 64 or dims(shared.w, 0) != N or dims(shared.w, 1) != K / 4 or dims(shared.s, 1) != K / 32) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ ci(R), HC, ci(N) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, bfd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(N / rb * nt), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(nt), 1, 1));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "ROWS", R }, .{ "RB", rb }, .{ "HC", HC }, .{ "TOPK", TOPK } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", bfd));
    const ins = [_]A{ act, down.w, down.s, down.b, shared.w, shared.s, idx, x1, sp.post, sp.comb };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The rows a threadgroup owns on this GPU: `rows_per_tg`, halved until the GPU grants the threads of the six-row
/// instantiation (M1/M2 cap a register-heavy kernel below 1024) for `K` inputs and `N` outputs; `error.Declined` where
/// `moeDown` does not serve them.
pub fn probe(gpa: std.mem.Allocator, kern: mlx.mlx_fast_metal_kernel, K: usize, N: usize) !usize {
    var rb: usize = rows_per_tg;
    while (true) {
        probeOnce(gpa, kern, rb, K, N) catch |e| {
            if (e != error.MlxError or rb <= 8 or !mlx.takeErrorIf("Thread group size")) return e;
            rb /= 2;
            continue;
        };
        return rb;
    }
}

fn probeOnce(gpa: std.mem.Allocator, kern: mlx.mlx_fast_metal_kernel, rb: usize, K: usize, N: usize) !void {
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const down: dsv41.Bank = .{ .w = try sc.zeros(&.{ 1, ci(N), ci(K * 3 / 32) }, .uint32), .s = try sc.zeros(&.{ 1, ci(N), ci(K / 64) }, .bfloat16), .b = try sc.zeros(&.{ 1, ci(N), ci(K / 64) }, .bfloat16), .qp = .{ .bits = 3, .group_size = 64 } };
    const shared: dsv41.Lin = .{ .w = try sc.zeros(&.{ ci(N), ci(K / 4) }, .uint32), .s = try sc.zeros(&.{ ci(N), ci(K / 32) }, .uint8), .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
    const R = decode.max_rows;
    const sp: dsv41.Split = .{ .pre = none, .post = try sc.zeros(&.{ R, HC }, .float32), .comb = try sc.zeros(&.{ R, HC, HC }, .float32) };
    const y = (try moeDown(&sc, kern, rb, &down, &shared, try sc.zeros(&.{ R, TOPK + 1, ci(K) }, .bfloat16), try sc.zeros(&.{ R, TOPK }, .uint32), try sc.zeros(&.{ R, HC, ci(N) }, .bfloat16), &sp)) orelse return error.Declined;
    try mlx.check(mlx.mlx_array_eval(y));
}

// ── tests ──

const fixture = @import("fixture.zig");

/// `rows` rows of `TOPK` distinct experts out of `n`; with `pool` > 0 each row's first three picks come from the first
/// `pool` experts, which every row shares.
fn picks(rng: std.Random, rows: usize, n: u32, pool: u32, out: []u32) void {
    for (0..rows) |r| {
        const row = out[r * TOPK ..][0..TOPK];
        for (row, 0..) |*e, j| while (true) {
            e.* = if (j < 3 and pool > 0) rng.uintLessThan(u32, pool) else rng.uintLessThan(u32, n);
            if (std.mem.indexOfScalar(u32, row[0..j], e.*) == null) break;
        };
    }
}

fn dequant(sc: *dsv41.Scope, w: A, s: A, b: A, bits: u32, gs: u32, mode: [*:0]const u8) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_dequantize(&o, w, s, b, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), mode, none, .{ .value = .float32, .has_value = true }, sc.s), &o);
}

/// The replaced chain at `rows` rows: R = 1 the fused down kernel `ffnDecode` runs; otherwise the down gather, its f32
/// sum, the shared down and `hcExpand`.
fn chainOf(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, act: A, idx: A, x1: A, sp: *const dsv41.Split) !A {
    const R = dims(act, 0);
    const K = dims(act, 2);
    const N = dims(x1, 2);
    const sdn = &ly.shared[2];
    if (R == 1) {
        const vdn = if (ly.experts.views) |v| v[1] else ly.experts.down;
        const sdb = if (sdn.b.ctx == null) sdn.s else sdn.b;
        return dsv41.moeKernel(sc, m.moe_dn_kernel, &.{ try sc.reshape(act, &.{ TOPK + 1, ci(K) }), vdn.w, vdn.s, vdn.b, sdn.w, sdn.s, sdb, try sc.reshape(idx, &.{TOPK}), x1, sp.post, sp.comb }, &.{ 1, HC, ci(N) }, .bfloat16, ly.experts.down.qp, sdn.qp, K, N, TOPK);
    }
    const dn = try dsv41.gatherMm(sc, try sc.reshape(try sc.slice(act, 1, 0, TOPK), &.{ ci(R), TOPK, 1, ci(K) }), &ly.experts.down, none, idx, false);
    const routed = try sc.sum(try sc.astype(try sc.reshape(dn, &.{ ci(R), TOPK, ci(N) }), .float32), 1, false);
    const shared = try sdn.apply(sc, try sc.reshape(try sc.slice(act, 1, TOPK, TOPK + 1), &.{ ci(R), ci(K) }));
    return dsv41.hcExpand(sc, x1, try dsv41.addShared(sc, m, routed, shared), sp);
}

/// f32 truth over the dequantized weights: y = sum of the picked experts' rows times their activations, plus the shared
/// row's; then the expand in f32.
fn truthOf(sc: *dsv41.Scope, ly: *const dsv41.Layer, act: A, picked: []const u32, x1: A, sp: *const dsv41.Split) !A {
    const R = dims(act, 0);
    const K = dims(act, 2);
    const N = dims(x1, 2);
    const bank = &ly.experts.down;
    const dq = try dequant(sc, bank.w, bank.s, bank.b, bank.qp.bits, bank.qp.group_size, "affine");
    const dsh = try dequant(sc, ly.shared[2].w, ly.shared[2].s, none, 8, 32, "mxfp8");
    const act32 = try sc.astype(act, .float32);
    var ys: [decode.max_rows]A = undefined;
    for (0..R) |r| {
        const sel = try sc.take(dq, try sc.keep(mlx.mlx_array_new_data(picked[r * TOPK ..].ptr, &[_]c_int{TOPK}, 1, .uint32)), 0);
        const a = try sc.reshape(try sc.slice(try sc.slice(act32, 0, r, r + 1), 1, 0, TOPK), &.{ TOPK, ci(K), 1 });
        const routed = try sc.sum(try sc.matmul(sel, a), 0, false);
        const sh = try sc.matmul(dsh, try sc.reshape(try sc.slice(try sc.slice(act32, 0, r, r + 1), 1, TOPK, TOPK + 1), &.{ ci(K), 1 }));
        ys[r] = try sc.reshape(try sc.add(routed, sh), &.{ 1, ci(N) });
    }
    const y = try sc.concat(ys[0..R], 0);
    const mixed = try sc.matmul(try sc.transpose(sp.comb, &.{ 0, 2, 1 }), try sc.astype(x1, .float32));
    return sc.add(try sc.mul(try sc.expand(sp.post, 2), try sc.expand(y, 1)), mixed);
}

/// The new kernel against today's chain, both against f32 truth, at `rows` rows of layer `li`.
fn checkRows(gpa: std.mem.Allocator, fx: *fixture.Fixture, li: usize, kern: mlx.mlx_fast_metal_kernel, rb: usize, rows: usize, pool: u32) !void {
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const m = fx.m;
    const ly = fx.layer(li);
    const K = m.cfg.moe_intermediate_size;
    const N = m.cfg.hidden_size;
    const act = try sc.keep(try fx.normal(&.{ ci(rows), TOPK + 1, ci(K) }, .bfloat16, 0, 0.5));
    const x1 = try sc.keep(try fx.stream(rows));
    const sp: dsv41.Split = .{ .pre = none, .post = try sc.keep(try fx.uniform(&.{ ci(rows), HC }, 0, 2)), .comb = try sc.keep(try fx.uniform(&.{ ci(rows), HC, HC }, 0, 0.5)) };
    var pk: [decode.max_rows * TOPK]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(li * 64 + rows);
    picks(prng.random(), rows, m.cfg.num_experts, pool, &pk);
    const idx = try sc.keep(mlx.mlx_array_new_data(&pk, &[_]c_int{ ci(rows), TOPK }, 2, .uint32));
    const down: *const dsv41.Bank = if (ly.experts.views) |*v| &v[1] else &ly.experts.down;
    const got = (try moeDown(&sc, kern, rb, down, &ly.shared[2], act, idx, x1, &sp)) orelse return error.Declined;
    const g = try dsv41.hostF32(got);
    defer gpa.free(g);
    const c = try dsv41.hostF32(try chainOf(&sc, m, ly, act, idx, x1, &sp));
    defer gpa.free(c);
    const t = try dsv41.hostF32(try truthOf(&sc, ly, act, &pk, x1, &sp));
    defer gpa.free(t);
    var label: [96]u8 = undefined;
    try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_down layer {d} R={d} pool={d}", .{ li, rows, pool }), g, c, t);
    // A block's row is the one-row launch's, to the order its sums run in.
    for (0..rows) |r| {
        const one = (try moeDown(&sc, kern, rb, down, &ly.shared[2], try sc.slice(act, 0, r, r + 1), try sc.slice(idx, 0, r, r + 1), try sc.slice(x1, 0, r, r + 1), &.{ .pre = none, .post = try sc.slice(sp.post, 0, r, r + 1), .comb = try sc.slice(sp.comb, 0, r, r + 1) })) orelse return error.Declined;
        const o = try dsv41.hostF32(one);
        defer gpa.free(o);
        try std.testing.expect(decode.relRms(g[r * HC * N ..][0 .. HC * N], o) < 5e-3);
    }
}

test "dsv41 dec moe_down: 1 to 6 rows are no further from f32 truth than today's chain, distinct and shared picks" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    mlx.installErrorHandler();
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 } }) |range| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = range[0], .hi = range[1], .ctx = 0 });
        defer fx.deinit();
        const rb = try probe(gpa, kern, fx.m.cfg.moe_intermediate_size, fx.m.cfg.hidden_size);
        for (range[0]..range[1]) |li| for ([_]usize{ 1, 2, 3, 4, 5, 6 }) |rows| {
            // Three and five rows once a range: the job tables split their pairs at 32 (30 stop short of it, 36 cross it).
            if ((rows == 3 or rows == 5) and li != range[0]) continue;
            for ([_]u32{ 0, 8 }) |pool| try checkRows(gpa, fx, li, kern, rb, rows, pool);
        };
    }
}

fn serve(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, rb: usize, o: struct { rows: usize = 2, bits: u32 = 3, shared_mode: params.QuantMode = .mxfp8, x_dt: mlx.mlx_dtype = .bfloat16, k: usize = 256 }) !?A {
    const K = o.k;
    const N = 64;
    const down: dsv41.Bank = .{ .w = try sc.zeros(&.{ 1, N, ci(K * o.bits / 32) }, .uint32), .s = try sc.zeros(&.{ 1, N, ci(K / 64) }, .bfloat16), .b = try sc.zeros(&.{ 1, N, ci(K / 64) }, .bfloat16), .qp = .{ .bits = o.bits, .group_size = 64 } };
    const shared: dsv41.Lin = .{ .w = try sc.zeros(&.{ N, ci(K / 4) }, .uint32), .s = try sc.zeros(&.{ N, ci(K / 32) }, .uint8), .qp = .{ .bits = 8, .group_size = 32, .mode = o.shared_mode } };
    const R = ci(o.rows);
    const sp: dsv41.Split = .{ .pre = none, .post = try sc.zeros(&.{ R, HC }, .float32), .comb = try sc.zeros(&.{ R, HC, HC }, .float32) };
    return moeDown(sc, kern, rb, &down, &shared, try sc.zeros(&.{ R, TOPK + 1, ci(K) }, .bfloat16), try sc.zeros(&.{ R, TOPK }, .uint32), try sc.zeros(&.{ R, HC, N }, o.x_dt), &sp);
}

test "dsv41 dec moe_down: declines what it does not serve and runs what it does" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const y = (try serve(&sc, kern, 32, .{})) orelse return error.Declined;
    try mlx.check(mlx.mlx_array_eval(y));
    try std.testing.expect((try serve(&sc, kern, 32, .{ .rows = decode.max_rows + 1 })) == null);
    try std.testing.expect((try serve(&sc, kern, 32, .{ .bits = 4 })) == null);
    try std.testing.expect((try serve(&sc, kern, 32, .{ .shared_mode = .affine })) == null);
    try std.testing.expect((try serve(&sc, kern, 32, .{ .x_dt = .float32 })) == null);
    try std.testing.expect((try serve(&sc, kern, 32, .{ .k = 192 })) == null);
    try std.testing.expect((try serve(&sc, kern, 128, .{})) == null);
}

// ── ubench ──

const Rng = struct {
    seed: u64,

    fn key(self: *Rng) !A {
        self.seed += 1;
        var k = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&k, self.seed));
        return k;
    }

    /// Random words (any bits are valid 3-bit codes and fp8 bytes bar fp8's NaNs, which the kernel reads as 480).
    fn bits(self: *Rng, sc: *dsv41.Scope, shape: []const c_int) !A {
        const k = try self.key();
        defer _ = mlx.mlx_array_free(k);
        var o = mlx.mlx_array_new();
        const a = try sc.res(mlx.mlx_random_bits(&o, shape.ptr, shape.len, 4, k, sc.s), &o);
        try mlx.check(mlx.mlx_array_eval(a));
        return a;
    }

    fn normal(self: *Rng, sc: *dsv41.Scope, shape: []const c_int, std_: f32) !A {
        const k = try self.key();
        defer _ = mlx.mlx_array_free(k);
        var o = mlx.mlx_array_new();
        const a = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .bfloat16, 0, std_, k, sc.s), &o);
        try mlx.check(mlx.mlx_array_eval(a));
        return a;
    }

    /// e8m0 scales around 1.
    fn exps(self: *Rng, sc: *dsv41.Scope, shape: []const c_int) !A {
        const k = try self.key();
        defer _ = mlx.mlx_array_free(k);
        const lo = try sc.keep(mlx.mlx_array_new_int(120));
        const hi = try sc.keep(mlx.mlx_array_new_int(128));
        var o = mlx.mlx_array_new();
        const a = try sc.res(mlx.mlx_random_randint(&o, lo, hi, shape.ptr, shape.len, .uint8, k, sc.s), &o);
        try mlx.check(mlx.mlx_array_eval(a));
        return a;
    }
};

/// Microseconds a step of a dependent chain on the GPU: the n-step graph is built untimed, then evaluated (the host's
/// graph building is not the kernel's).
fn evalUs(ctx: anytype, n: usize) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var last: A = .{ .ctx = null };
        for (0..n) |i| last = try ctx.step(i);
        const t0 = iou.nowUsMonotonic(io);
        try mlx.check(mlx.mlx_array_eval(last));
        const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / @as(f64, @floatFromInt(n));
        if (rep >= 2) times[rep - 2] = us;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

const Chain = struct {
    sc: *dsv41.Scope,
    kern: mlx.mlx_fast_metal_kernel,
    dn_kern: mlx.mlx_fast_metal_kernel,
    rb: usize,
    bank: *const dsv41.Bank,
    view: *const dsv41.Bank,
    shared: []const dsv41.Lin,
    act: A,
    idxs: []const A,
    post: A,
    comb: A,
    x: A,
    rows: usize,
    old: bool,

    /// One step of the chain: this layer's down half on the stream the last step made.
    pub fn step(self: *Chain, i: usize) !A {
        const sh = &self.shared[i % self.shared.len];
        const idx = self.idxs[i % self.idxs.len];
        const sp: dsv41.Split = .{ .pre = none, .post = self.post, .comb = self.comb };
        const K = dims(self.act, 2);
        const N = dims(self.x, 2);
        if (!self.old) {
            self.x = (try moeDown(self.sc, self.kern, self.rb, self.view, sh, self.act, idx, self.x, &sp)).?;
        } else if (self.rows == 1) {
            self.x = try dsv41.moeKernel(self.sc, self.dn_kern, &.{ try self.sc.reshape(self.act, &.{ TOPK + 1, ci(K) }), self.view.w, self.view.s, self.view.b, sh.w, sh.s, sh.s, try self.sc.reshape(idx, &.{TOPK}), self.x, self.post, self.comb }, &.{ 1, HC, ci(N) }, .bfloat16, self.bank.qp, sh.qp, K, N, TOPK);
        } else {
            const R = ci(self.rows);
            const dn = try dsv41.gatherMm(self.sc, try self.sc.reshape(try self.sc.slice(self.act, 1, 0, TOPK), &.{ R, TOPK, 1, ci(K) }), self.bank, none, idx, false);
            const routed = try self.sc.sum(try self.sc.astype(try self.sc.reshape(dn, &.{ R, TOPK, ci(N) }), .float32), 1, false);
            const shared = try sh.apply(self.sc, try self.sc.reshape(try self.sc.slice(self.act, 1, TOPK, TOPK + 1), &.{ R, ci(K) }));
            const y = try self.sc.astype(try self.sc.add(routed, try self.sc.astype(shared, .float32)), .bfloat16);
            self.x = try dsv41.hcExpand(self.sc, self.x, y, &sp);
        }
        return self.x;
    }
};

test "dsv41 dec moe_down: ubench" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    mlx.installErrorHandler();
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var rng: Rng = .{ .seed = 1 };
    const E = 384;
    const K = 2304;
    const N = 5120;
    // The release's routed down bank (1.98 GB of random 3-bit codes: reads come from DRAM) and 16 shared copies.
    const bank: dsv41.Bank = .{ .w = try rng.bits(&sc, &.{ E, N, K * 3 / 32 }), .s = try rng.normal(&sc, &.{ E, N, K / 64 }, 0.01), .b = try rng.normal(&sc, &.{ E, N, K / 64 }, 0.01), .qp = .{ .bits = 3, .group_size = 64 } };
    var view = bank;
    inline for (.{ "w", "s", "b" }) |f| {
        const a = @field(bank, f);
        const sh = mlx.getShape(a);
        var o = mlx.mlx_array_new();
        @field(view, f) = try sc.res(mlx.mlx_slice(&o, a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, sc.s), &o);
    }
    var shared: [16]dsv41.Lin = undefined;
    for (&shared) |*s| s.* = .{ .w = try rng.bits(&sc, &.{ N, K / 4 }), .s = try rng.exps(&sc, &.{ N, K / 32 }), .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const dn_kern = try dsv41.buildKernel("dsv41_moe_dn", &dsv41.MOE_DN_INS, &.{"out"}, dsv41.MOE_DN_SOURCE, dsv41.MOE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(dn_kern);
    const rb = try probe(gpa, kern, K, N);
    const steps = 200;
    // R = 1 (6 experts), R = 6 with every pick random (about 35 distinct experts) and R = 6 where each row's first three
    // picks come from 8 experts all rows share (about 25).
    for ([_][2]u32{ .{ 1, 0 }, .{ 6, 0 }, .{ 6, 8 } }) |cfg| {
        const rows: usize = cfg[0];
        var prng = std.Random.DefaultPrng.init(5);
        var idxs: [steps]A = undefined;
        var distinct: f64 = 0;
        for (&idxs) |*ix| {
            var pk: [decode.max_rows * TOPK]u32 = undefined;
            picks(prng.random(), rows, E, cfg[1], &pk);
            var seen: [decode.max_rows * TOPK]u32 = undefined;
            var n: usize = 0;
            for (pk[0 .. rows * TOPK]) |e| if (std.mem.indexOfScalar(u32, seen[0..n], e) == null) {
                seen[n] = e;
                n += 1;
            };
            distinct += @floatFromInt(n);
            ix.* = try sc.keep(mlx.mlx_array_new_data(&pk, &[_]c_int{ ci(rows), TOPK }, 2, .uint32));
        }
        distinct /= steps;
        var pv: [decode.max_rows * HC]f32 = undefined;
        for (&pv, 0..) |*v, i| v.* = 0.4 + 0.05 * @as(f32, @floatFromInt(i % HC));
        var cv: [decode.max_rows * HC * HC]f32 = undefined;
        for (&cv, 0..) |*v, i| v.* = 0.2 + 0.01 * @as(f32, @floatFromInt(i % 7));
        var ctx: Chain = .{
            .sc = &sc,
            .kern = kern,
            .dn_kern = dn_kern,
            .rb = rb,
            .bank = &bank,
            .view = &view,
            .shared = &shared,
            .act = try rng.normal(&sc, &.{ ci(rows), TOPK + 1, K }, 0.5),
            .idxs = &idxs,
            .post = try sc.keep(mlx.mlx_array_new_data(&pv, &[_]c_int{ ci(rows), HC }, 2, .float32)),
            .comb = try sc.keep(mlx.mlx_array_new_data(&cv, &[_]c_int{ ci(rows), HC, HC }, 3, .float32)),
            .x = try rng.normal(&sc, &.{ ci(rows), HC, N }, 1),
            .rows = rows,
            .old = false,
        };
        const bytes = distinct * N * (K * 3 / 32 * 4 + 2 * K / 64 * 2) + N * (K + K / 32);
        const new_raw = try decode.chainUs(&ctx, steps);
        const new_eval = try evalUs(&ctx, steps);
        ctx.old = true;
        const old_raw = try decode.chainUs(&ctx, steps);
        const old_eval = try evalUs(&ctx, steps);
        std.debug.print("[dec-ubench] moe_down: {d:.1} distinct experts a step (picks pool {d})\n", .{ distinct, cfg[1] });
        decode.report("moe_down", rows, new_raw, bytes, old_raw);
        std.debug.print("[dec-ubench] moe_down eval-only (host graph building not counted): ", .{});
        decode.report("moe_down", rows, new_eval, bytes, old_eval);
    }
}

// ── the kernel contract ──

const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const contract = @import("../contract.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

const adt: mlx.mlx_dtype = .bfloat16;

/// fp32's unit roundoff.
const u = 0x1p-24;

fn bf(v: f64) f64 {
    return spec.roundTo(.bf16, v);
}

fn depthK(s: Shape) usize {
    return s.k;
}

const down_l = metal.Launch{ .name = "dsv41_dec_moe_down", .inputs = &INS, .outputs = &.{"out"}, .source = SOURCE, .header = HEADER, .tg_bytes = tgBytes(max_k, decode.max_rows), .m1 = 640, .m2 = 640 };

/// Row `d` of the shared mxfp8 operand as float64: the `words` (e4m3 bytes, little endian) times the e8m0 `exps`.
fn sharedAt(words: []const f64, exps: []const f64, d: usize, i: usize, k: usize) f64 {
    const word: u32 = @intFromFloat(words[d * (k / 4) + i / 4]);
    const byte: u8 = @truncate(word >> @intCast(8 * (i % 4)));
    return kc.host.e4m3(byte) * std.math.pow(f64, 2, exps[d * (k / 32) + i / 32] - 127);
}

/// The block's down half: `rows` decode rows over `Params.experts` routed experts of a 3-bit group-64 bank plus the
/// shared mxfp8 expert, y rounded to bf16 once, then the hc expand.
const DownK = struct {
    pub const Params = struct { experts: usize = 8 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        const nt = rows_per_tg * (s.k / 128);
        if (s.rows < 1 or s.rows > decode.max_rows or s.n == 0 or s.n % rows_per_tg != 0 or s.k == 0 or s.k % 128 != 0 or s.k > max_k or p.experts < TOPK or p.experts > 64) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.n / rows_per_tg * nt), 1, 1 }, .tg = .{ ci(nt), 1, 1 } };
        g.out(&.{ ci(s.rows), HC, ci(s.n) }, adt);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("ROWS", s.rows);
        g.int("RB", rows_per_tg);
        g.int("HC", HC);
        g.int("TOPK", TOPK);
        g.dtype("T", adt);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const p = k.p;
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const x = try k.add(adt, &.{ R, TOPK + 1, K });
        const routed = try contract.Bank.add(k, &.{ p.experts, N }, 3, 64, K);
        const words = try k.add(.uint32, &.{ N, K / 4 });
        const exps = try k.add(.uint8, &.{ N, K / 32 });
        const inds = try k.add(.uint32, &.{ R, TOPK });
        const res = try k.add(adt, &.{ R, HC, N });
        const post = try k.add(.float32, &.{ R, HC });
        const comb = try k.add(.float32, &.{ R, HC, HC });
        kit.uniform(rng, x, -2, 2);
        kit.uniform(rng, res, -3, 3);
        kit.uniform(rng, post, 0, 2);
        kit.uniform(rng, comb, 0, 0.5);
        // Distinct experts in a row; rows may share them.
        for (0..R) |r| {
            var used: [64]bool = @splat(false);
            for (inds[r * TOPK ..][0..TOPK]) |*v| {
                var e = rng.uintLessThan(usize, p.experts);
                while (used[e]) e = (e + 1) % p.experts;
                used[e] = true;
                v.* = @floatFromInt(e);
            }
        }
        const bytes = try k.scratch().alloc(u8, N * K);
        defer k.scratch().free(bytes);
        switch (f) {
            .random, .nonfinite => {
                try routed.random(rng, k.scratch());
                for (bytes) |*b| {
                    b.* = rng.int(u8);
                    if (b.* & 0x7f == 0x7f) b.* -= 1; // S.1111.111 is fp8's NaN; the kernel reads it as 480
                }
                for (exps) |*e| e.* = @floatFromInt(rng.intRangeAtMost(u8, 120, 130));
                if (f == .nonfinite) {
                    // A NaN or infinite group scale in output s's row of row 0's slot s (s < 4): NaN, +inf, -inf, +inf. Every
                    // pair's first group is positive, so a row sharing the expert meets the same infinity.
                    const put = [_]f64{ kit.nan, kit.inf, -kit.inf, kit.inf };
                    for (0..R * (TOPK + 1)) |i| kit.uniform(rng, x[i * K ..][0..64], 0.1, 2);
                    for (put, 0..) |v, s| try routed.plant(@as(usize, @intFromFloat(inds[s])) * N + s, v, k.scratch());
                }
            },
            .stress => {
                // Every pair's activations the same mirrored halves; post 1, comb 0: the output is the summed dots, exact.
                const perm = try k.scratch().alloc(usize, K / 2);
                defer k.scratch().free(perm);
                contract.mirroredX(rng, x[0..K], perm);
                for (1..R * (TOPK + 1)) |i| @memcpy(x[i * K ..][0..K], x[0..K]);
                try routed.mirrored(rng, perm, k.scratch());
                // fp8 integers 1..15 (exact) on the first half, their negatives on the shuffled second, unit scales.
                for (0..N) |d| for (0..K / 2) |i| {
                    const c: u8 = if (i < 8) 1 else rng.intRangeAtMost(u8, 1, 15);
                    var code: u8 = 0;
                    while (kc.host.e4m3(code) != @as(f64, @floatFromInt(c))) code += 1;
                    bytes[d * K + i] = code;
                    bytes[d * K + K / 2 + perm[i]] = code | 0x80;
                };
                @memset(exps, 127);
                @memset(post, 1);
                @memset(comb, 0);
            },
        }
        for (words, 0..) |*wd, i| wd.* = @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * HC * s.n;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const routed = contract.Bank{ .w = k.in(1), .s = k.in(2), .b = k.in(3), .bits = 3, .gs = 64, .k = K };
        const t = try k.scratch().alloc(f64, K);
        defer k.scratch().free(t);
        // y's one bf16 rounding, as a sum error: gamma(K) of the terms' magnitude covers an eighth-bit of the sum.
        const flip = 0x1p-8 / spec.gamma(@floatFromInt(K), u);
        var parts: [TOPK + 1]f64 = undefined;
        for (0..R) |r| for (0..N) |d| {
            var m: f64 = 0;
            for (0..TOPK) |s| {
                const e: usize = @intFromFloat(k.in(6)[r * TOPK + s]);
                const q = routed.dot(e * N + d, k.in(0)[(r * (TOPK + 1) + s) * K ..][0..K], sum, t);
                parts[s] = q[0];
                m += q[1];
            }
            const xs = k.in(0)[(r * (TOPK + 1) + TOPK) * K ..][0..K];
            var sm: f64 = 0;
            for (0..K) |i| {
                const term = xs[i] * sharedAt(k.in(4), k.in(5), d, i, K);
                t[i] = term;
                sm += @abs(term);
            }
            parts[TOPK] = sum.of(t[0..K]);
            m += sm;
            const tot = sum.of(&parts);
            const y = bf(tot);
            m += (@abs(tot) + @abs(y)) * flip;
            for (0..HC) |i| {
                const po = k.in(8)[r * HC + i];
                var acc = po * y;
                var cm: f64 = 0;
                for (0..HC) |j| {
                    const term = k.in(9)[(r * HC + j) * HC + i] * k.in(7)[(r * HC + j) * N + d];
                    acc += term;
                    cm += @abs(term);
                }
                out[(r * HC + i) * N + d] = acc;
                mag[(r * HC + i) * N + d] = @abs(po) * m + cm;
            }
        };
    }
};

/// Served shapes cut the output rows to a few where the float64 reference would run for seconds; the reduction axis stays
/// the release's.
pub fn register(set: *kc.Set) !void {
    try kit.add(set, DownK, &down_l, .{}, .{
        .name = "dsv41 dec moe_down (3-bit g64 routed + mxfp8 shared, hc expand)",
        .accum = .f32,
        .depth = depthK,
        .ops = 4,
        .mids = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = rows_per_tg, .step = rows_per_tg, .min = rows_per_tg }, .{ .size = 128, .step = 128, .min = 128 } },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 2304 }, .{ .rows = 6, .n = 64, .k = 2304 }, .{ .rows = 2, .n = 32, .k = 128 } },
    });
}

test "dsv41 dec moe_down: the declared launch is the shipped wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    for ([_]usize{ 1, 2, 6 }) |rows| {
        const c = try a.create(kit.Case(DownK));
        defer {
            c.deinit();
            a.destroy(c);
        }
        c.* = .init(a, &down_l, .{});
        try std.testing.expect(try c.prepare(.{ .rows = rows, .n = 64, .k = 256 }, .random, 7));
        try c.run();
        const mine = try a.dupe(u8, c.rig.outputBytes());
        defer a.free(mine);
        var sc = dsv41.Scope.init(a, c.rig.stream());
        defer sc.deinit();
        const down: dsv41.Bank = .{ .w = c.args[1], .s = c.args[2], .b = c.args[3], .qp = .{ .bits = 3, .group_size = 64 } };
        const shared: dsv41.Lin = .{ .w = c.args[4], .s = c.args[5], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
        const got = (try moeDown(&sc, kern, rows_per_tg, &down, &shared, c.args[0], c.args[6], c.args[7], &.{ .pre = none, .post = c.args[8], .comb = c.args[9] })) orelse return error.Declined;
        try std.testing.expectEqualSlices(u8, try c.rig.bytesOf(&.{got}), mine);
    }
}

test "dsv41 dec moe_down: the registered kernel passes the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
}
