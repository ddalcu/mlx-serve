//! Env-gated microbench for the two fused decode MoE kernels at Qwen3.8-Flash-Next's
//! shapes (E 512, top-10, hidden 2560, intermediate 640, 4-bit affine g64, bf16):
//! `gatherQmvGateUp` and `gatherQmvDownReduce`, against MLX's stock `gather_qmm`
//! chains, so a kernel variant is timed in isolation instead of behind a server boot.
//!   MOE_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="moe gather kernels"

const std = @import("std");
const mlx = @import("mlx.zig");
const xfm = @import("transformer.zig");
const io_util = @import("io_util.zig");

const E: c_int = 512;
const TOPK: c_int = 10;
const H: c_int = 2560;
const I: c_int = 640;
const BITS: u32 = 4;
const GS: u32 = 64;
const WARM = 5;
const BATCHES = 20;
const CALLS_SMALL = 10;
const MAX_CALLS = 40;

const Bank = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,

    fn deinit(b: Bank) void {
        _ = mlx.mlx_array_free(b.w);
        _ = mlx.mlx_array_free(b.sc);
        _ = mlx.mlx_array_free(b.bi);
    }

    /// Bytes one call reads from this bank: TOPK rows of packed words + bf16 scale and bias per group.
    fn bytesPerCall(n: c_int, k: c_int) f64 {
        const words = @divExact(@as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(BITS)), 32.0);
        const groups = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(GS));
        return @as(f64, @floatFromInt(TOPK)) * @as(f64, @floatFromInt(n)) * (words * 4.0 + groups * 4.0);
    }
};

fn randUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

/// Any bit pattern is a valid 4-bit row, so the words come straight from the RNG.
fn affineBank(rnd: std.Random, n: c_int, k: c_int, s: mlx.mlx_stream) !Bank {
    const words: c_int = @intCast(@divExact(@as(u32, @intCast(k)) * BITS, 32));
    const groups: c_int = @divExact(k, @as(c_int, @intCast(GS)));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const w_shape = [_]c_int{ E, n, words };
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &w_shape, 3, 4, key, s));
    const s_shape = [_]c_int{ E, n, groups };
    const sc = try randUniformBf16(rnd, &s_shape, 0.01, 0.06, s);
    const bi = try randUniformBf16(rnd, &s_shape, -0.5, 0.5, s);
    return .{ .w = w, .sc = sc, .bi = bi };
}

fn evalAll(outs: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |o| _ = mlx.mlx_vector_array_append_value(vec, o);
    try mlx.check(mlx.mlx_eval(vec));
}

fn allFinite(a: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    try mlx.check(mlx.mlx_isfinite(&fin, a, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, fin, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var v = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, all));
    return v;
}

const Inputs = struct {
    s: mlx.mlx_stream,
    x: mlx.mlx_array, // [H] bf16
    x_down: mlx.mlx_array, // [TOPK, I] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [TOPK], distinct experts each, cycled by call index
    lhs_zero: mlx.mlx_array, // uint32 [TOPK] zeros: every expert reads the one token
    lhs_iota: mlx.mlx_array, // uint32 [TOPK] 0..9: every expert reads its own row
    scores: mlx.mlx_array, // [TOPK] bf16, sums to ~1
};

const Arm = enum { gateup, downred, pair, stock_gateup, stock_downred };

fn stockGateUp(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x, &.{ 1, 1, H }, 3, s));
    var xg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xg);
    try mlx.check(mlx.mlx_broadcast_to(&xg, x3, &.{ TOPK, 1, H }, 3, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_gather_qmm(&g, xg, in.gate.w, in.gate.sc, in.gate.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_gather_qmm(&u, xg, in.up.w, in.up.sc, in.up.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_multiply(&out, silu, u, s));
    return out;
}

fn stockDownReduce(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x_down, &.{ TOPK, 1, I }, 3, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_gather_qmm(&d, x3, in.down.w, in.down.sc, in.down.bi, in.lhs_iota, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, in.scores, &.{ TOPK, 1, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

/// One call of `arm`. `pair` is the in-situ per-layer chain: gate+up, then down+reduce over
/// the activation it produced, so its two dispatches are dependent like a real layer's.
fn call(in: *const Inputs, arm: Arm, i: usize) !mlx.mlx_array {
    const inds = in.inds[i % CALLS_SMALL];
    switch (arm) {
        .gateup => return (try xfm.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse error.GateUpDeclined,
        .downred => return (try xfm.gatherQmvDownReduce(in.s, in.x_down, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined,
        .pair => {
            const act = (try xfm.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse return error.GateUpDeclined;
            defer _ = mlx.mlx_array_free(act);
            return (try xfm.gatherQmvDownReduce(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined;
        },
        .stock_gateup => return stockGateUp(in, inds),
        .stock_downred => return stockDownReduce(in, inds),
    }
}

/// Returns the sorted wall times of BATCHES batches of `n` independent calls (distinct index
/// vectors per call, one eval per batch), after WARM warm-up batches.
fn timeBatches(in: *const Inputs, arm: Arm, n: usize, out: *[BATCHES]u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var outs: [MAX_CALLS]mlx.mlx_array = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        for (outs[0..n], 0..) |*o, i| o.* = try call(in, arm, i);
        try evalAll(outs[0..n]);
        if (b >= WARM) out[b - WARM] = sw.read();
        for (outs[0..n]) |o| _ = mlx.mlx_array_free(o);
    }
    std.mem.sort(u64, out, {}, std.sort.asc(u64));
}

/// Prints the per-call time at CALLS_SMALL calls per batch and the MARGINAL per-call time (the
/// slope between CALLS_SMALL and MAX_CALLS calls per batch, which cancels the fixed eval/submit
/// cost each batch pays once), with the GB/s the marginal implies from the bank bytes a call reads.
fn bench(in: *const Inputs, arm: Arm, label: []const u8, bytes_per_call: f64) !void {
    var small: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, CALLS_SMALL, &small);
    var large: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, MAX_CALLS, &large);
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_us = @as(f64, @floatFromInt(small[BATCHES / 2])) / @as(f64, CALLS_SMALL) / 1000.0;
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-ubench] {s:<22} per-call@{d} {d:6.1} us | marginal median {d:6.1} us  min {d:6.1} us  ({d:5.1} MB/call -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        label, CALLS_SMALL, med_us, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// Bar: both fused kernels run at the Flash Next shapes and return finite outputs; timings are printed.
test "moe gather kernels: microbench (MOE_UBENCH=1)" {
    if (std.c.getenv("MOE_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };
    xfm.gqmv_gateup_override = true;
    defer xfm.gqmv_gateup_override = null;
    xfm.downred_override = true;
    defer xfm.downred_override = null;

    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x30E0B3);
    const rnd = prng.random();

    var in: Inputs = undefined;
    in.s = s;
    in.x = try randUniformBf16(rnd, &.{H}, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(in.x);
    in.x_down = try randUniformBf16(rnd, &.{ TOPK, I }, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(in.x_down);
    in.gate = try affineBank(rnd, I, H, s);
    defer in.gate.deinit();
    in.up = try affineBank(rnd, I, H, s);
    defer in.up.deinit();
    in.down = try affineBank(rnd, H, I, s);
    defer in.down.deinit();

    const ish = [_]c_int{TOPK};
    var made: usize = 0;
    defer for (in.inds[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&in.inds) |*a| {
        // Partial Fisher-Yates: TOPK distinct experts per vector, a fresh set per call.
        var pool: [E]u32 = undefined;
        for (&pool, 0..) |*p, k| p.* = @intCast(k);
        var pick: [TOPK]u32 = undefined;
        for (&pick, 0..) |*p, k| {
            const j = k + rnd.uintLessThan(usize, pool.len - k);
            std.mem.swap(u32, &pool[k], &pool[j]);
            p.* = pool[k];
        }
        a.* = mlx.mlx_array_new_data(&pick, &ish, 1, .uint32);
        made += 1;
    }
    const zeros = std.mem.zeroes([TOPK]u32);
    in.lhs_zero = mlx.mlx_array_new_data(&zeros, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(in.lhs_zero);
    var iota: [TOPK]u32 = undefined;
    for (&iota, 0..) |*v, k| v.* = @intCast(k);
    in.lhs_iota = mlx.mlx_array_new_data(&iota, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(in.lhs_iota);
    {
        var sc: [TOPK]f32 = undefined;
        var sum: f32 = 0;
        for (&sc) |*v| {
            v.* = 0.05 + rnd.float(f32);
            sum += v.*;
        }
        for (&sc) |*v| v.* /= sum;
        const sc32 = mlx.mlx_array_new_data(&sc, &ish, 1, .float32);
        defer _ = mlx.mlx_array_free(sc32);
        in.scores = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&in.scores, sc32, .bfloat16, s));
    }
    defer _ = mlx.mlx_array_free(in.scores);
    try evalAll(&.{ in.x, in.x_down, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, in.down.w, in.down.sc, in.down.bi, in.scores });

    // Sanity bar: finite outputs from both fused kernels.
    {
        const a = try call(&in, .gateup, 0);
        defer _ = mlx.mlx_array_free(a);
        try std.testing.expect(try allFinite(a, s));
        const d = try call(&in, .downred, 0);
        defer _ = mlx.mlx_array_free(d);
        try std.testing.expect(try allFinite(d, s));
    }

    const gateup_bytes = 2.0 * Bank.bytesPerCall(I, H);
    const down_bytes = Bank.bytesPerCall(H, I);
    std.debug.print("\n[moe-ubench] E={d} topk={d} hidden={d} inter={d} {d}-bit g{d} bf16; {d} batches each of {d} and {d} independent calls, one eval per batch\n", .{ E, TOPK, H, I, BITS, GS, BATCHES, CALLS_SMALL, MAX_CALLS });
    try bench(&in, .gateup, "gatherQmvGateUp", gateup_bytes);
    try bench(&in, .downred, "gatherQmvDownReduce", down_bytes);
    try bench(&in, .pair, "gateup -> downred", gateup_bytes + down_bytes);
    try bench(&in, .stock_gateup, "stock gather_qmm g+u", gateup_bytes);
    try bench(&in, .stock_downred, "stock gather_qmm down", down_bytes);
    try std.testing.expect(!mlx.errorPending());
}

// ── Verify-width rows arm ──────────────────────────────────────────────────────────────────────
// A solo verify of S rows reaches the MoE through `gatherQmvGateUpRows` + `gatherQmvDownReduceRows`
// (S <= 8). Bytes are counted over the DISTINCT experts the S rows pick, since rows that share an
// expert read its weights once.

const RowsIn = struct {
    s: mlx.mlx_stream,
    rows: c_int,
    x: mlx.mlx_array, // [S, H] bf16
    x_act: mlx.mlx_array, // [S, TOPK, I] bf16, the activation the down kernel reads
    scores: mlx.mlx_array, // [S, TOPK] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [S, TOPK], TOPK distinct experts per row
    distinct_avg: f64, // mean distinct experts per index table over the cycled tables
};

const RowsArm = enum { gateup, downred, pair, decode_pair_x_rows };

fn rowsCall(in: *const RowsIn, arm: RowsArm, i: usize) !mlx.mlx_array {
    const inds = in.inds[i % CALLS_SMALL];
    switch (arm) {
        .gateup => return (try xfm.gatherQmvGateUpRows(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine)) orelse error.GateUpRowsDeclined,
        .downred => return (try xfm.gatherQmvDownReduceRows(in.s, in.x_act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownRowsDeclined,
        .pair => {
            const act = (try xfm.gatherQmvGateUpRows(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine)) orelse return error.GateUpRowsDeclined;
            defer _ = mlx.mlx_array_free(act);
            return (try xfm.gatherQmvDownReduceRows(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownRowsDeclined;
        },
        // The same S tokens through the one-token decode kernels, one pair per row, as a solo verify would
        // run them without the rows arm. Each row's own [1, TOPK] expert vector is a slice of the table.
        .decode_pair_x_rows => {
            var acc = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(acc);
            var r: c_int = 0;
            while (r < in.rows) : (r += 1) {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_slice(&xr, in.x, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, H }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var x1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x1);
                try mlx.check(mlx.mlx_reshape(&x1, xr, &[_]c_int{H}, 1, in.s));
                var ir = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ir);
                try mlx.check(mlx.mlx_slice(&ir, inds, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, TOPK }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var ind1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ind1);
                try mlx.check(mlx.mlx_reshape(&ind1, ir, &[_]c_int{TOPK}, 1, in.s));
                var sr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sr);
                try mlx.check(mlx.mlx_slice(&sr, in.scores, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, TOPK }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var s1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(s1);
                try mlx.check(mlx.mlx_reshape(&s1, sr, &[_]c_int{TOPK}, 1, in.s));
                const act = (try xfm.gatherQmvGateUp(in.s, x1, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, ind1, BITS, GS, .affine, 0)) orelse return error.GateUpDeclined;
                defer _ = mlx.mlx_array_free(act);
                const d = (try xfm.gatherQmvDownReduce(in.s, act, in.down.w, in.down.sc, in.down.bi, ind1, s1, BITS, GS, .affine)) orelse return error.DownReduceDeclined;
                if (r == 0) {
                    _ = mlx.mlx_array_free(acc);
                    acc = d;
                } else {
                    _ = mlx.mlx_array_free(d);
                }
            }
            return acc;
        },
    }
}

fn timeRowsBatches(in: *const RowsIn, arm: RowsArm, n: usize, out: *[BATCHES]u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var outs: [MAX_CALLS]mlx.mlx_array = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        for (outs[0..n], 0..) |*o, i| o.* = try rowsCall(in, arm, i);
        try evalAll(outs[0..n]);
        if (b >= WARM) out[b - WARM] = sw.read();
        for (outs[0..n]) |o| _ = mlx.mlx_array_free(o);
    }
    std.mem.sort(u64, out, {}, std.sort.asc(u64));
}

fn benchRows(in: *const RowsIn, arm: RowsArm, label: []const u8, bytes_per_call: f64) !void {
    var small: [BATCHES]u64 = undefined;
    try timeRowsBatches(in, arm, CALLS_SMALL, &small);
    var large: [BATCHES]u64 = undefined;
    try timeRowsBatches(in, arm, MAX_CALLS, &large);
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-rows-ubench] S={d} {s:<22} marginal median {d:7.1} us  min {d:7.1} us  ({d:5.1} MB distinct -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        in.rows, label, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// The same rows pair, but each call's input depends on the previous call's output (through a zero that the
// graph cannot fold), so the calls cannot overlap: the shape of a forward, where a kernel's ramp-up and tail
// are on the critical path. The three small ops that carry the dependency are the same in every variant.
const ChainMode = enum { pair, gateup, downred, ops };

fn chainBatch(in: *const RowsIn, n: usize, mode: ChainMode) !void {
    var prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev);
    var have = false;
    for (0..n) |i| {
        // The array this call's input is perturbed through: x for gate+up and the pair, x_act for down.
        const base = if (mode == .downred) in.x_act else in.x;
        var xk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xk);
        if (have) {
            var sm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sm);
            try mlx.check(mlx.mlx_sum(&sm, prev, false, in.s));
            const zf = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zf);
            var z = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(z);
            try mlx.check(mlx.mlx_multiply(&z, sm, zf, in.s));
            var zb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zb);
            try mlx.check(mlx.mlx_astype(&zb, z, .bfloat16, in.s));
            try mlx.check(mlx.mlx_add(&xk, base, zb, in.s));
        } else {
            try mlx.check(mlx.mlx_array_set(&xk, base));
        }
        const inds = in.inds[i % CALLS_SMALL];
        var out = mlx.mlx_array_new();
        switch (mode) {
            .ops => try mlx.check(mlx.mlx_array_set(&out, xk)),
            .gateup => {
                _ = mlx.mlx_array_free(out);
                out = (try xfm.gatherQmvGateUpRows(in.s, xk, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine)) orelse return error.GateUpRowsDeclined;
            },
            .downred => {
                _ = mlx.mlx_array_free(out);
                out = (try xfm.gatherQmvDownReduceRows(in.s, xk, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse return error.DownRowsDeclined;
            },
            .pair => {
                const act = (try xfm.gatherQmvGateUpRows(in.s, xk, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine)) orelse return error.GateUpRowsDeclined;
                defer _ = mlx.mlx_array_free(act);
                _ = mlx.mlx_array_free(out);
                out = (try xfm.gatherQmvDownReduceRows(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse return error.DownRowsDeclined;
            },
        }
        _ = mlx.mlx_array_free(prev);
        prev = out;
        have = true;
    }
    try mlx.check(mlx.mlx_array_eval(prev));
}

fn benchChain(in: *const RowsIn, mode: ChainMode, label: []const u8, bytes_per_call: f64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var small: [BATCHES]u64 = undefined;
    var large: [BATCHES]u64 = undefined;
    for ([_]usize{ CALLS_SMALL, MAX_CALLS }, [_]*[BATCHES]u64{ &small, &large }) |n, dst| {
        for (0..WARM + BATCHES) |b| {
            var sw = io_util.Stopwatch.init(io);
            try chainBatch(in, n, mode);
            if (b >= WARM) dst[b - WARM] = sw.read();
        }
        std.mem.sort(u64, dst, {}, std.sort.asc(u64));
    }
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-rows-ubench] S={d} {s:<22} marginal median {d:7.1} us  min {d:7.1} us  ({d:5.1} MB distinct -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        in.rows, label, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

test "moe verify rows kernels: microbench at S rows (MOE_UBENCH=1)" {
    if (std.c.getenv("MOE_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };
    xfm.gqmv_gateup_override = true;
    defer xfm.gqmv_gateup_override = null;
    xfm.downred_override = true;
    defer xfm.downred_override = null;
    xfm.moe_rows_fused_override = true;
    defer xfm.moe_rows_fused_override = null;

    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x30E0B4);
    const rnd = prng.random();
    const gate = try affineBank(rnd, I, H, s);
    defer gate.deinit();
    const up = try affineBank(rnd, I, H, s);
    defer up.deinit();
    const down = try affineBank(rnd, H, I, s);
    defer down.deinit();
    try evalAll(&.{ gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, down.w, down.sc, down.bi });

    const gu_per_expert = 2.0 * Bank.bytesPerCall(I, H) / @as(f64, TOPK);
    const dn_per_expert = Bank.bytesPerCall(H, I) / @as(f64, TOPK);
    std.debug.print("\n[moe-rows-ubench] E={d} topk={d} hidden={d} inter={d} {d}-bit g{d}; rows kernels against S one-token decode pairs; GB/s over the distinct experts of the S rows\n", .{ E, TOPK, H, I, BITS, GS });

    for ([_]c_int{ 2, 3, 4, 6, 8 }) |S| {
        var in: RowsIn = undefined;
        in.s = s;
        in.rows = S;
        in.gate = gate;
        in.up = up;
        in.down = down;
        in.x = try randUniformBf16(rnd, &.{ S, H }, -1.0, 1.0, s);
        defer _ = mlx.mlx_array_free(in.x);
        in.x_act = try randUniformBf16(rnd, &.{ S, TOPK, I }, -1.0, 1.0, s);
        defer _ = mlx.mlx_array_free(in.x_act);
        in.scores = try randUniformBf16(rnd, &.{ S, TOPK }, 0.05, 0.15, s);
        defer _ = mlx.mlx_array_free(in.scores);
        var made: usize = 0;
        defer for (in.inds[0..made]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var distinct_sum: f64 = 0;
        for (&in.inds) |*a| {
            var table: [8 * TOPK]u32 = undefined;
            var seen = std.mem.zeroes([E]bool);
            var distinct: u32 = 0;
            var r: usize = 0;
            while (r < @as(usize, @intCast(S))) : (r += 1) {
                var pool: [E]u32 = undefined;
                for (&pool, 0..) |*p, k| p.* = @intCast(k);
                var k: usize = 0;
                while (k < TOPK) : (k += 1) {
                    const j = k + rnd.uintLessThan(usize, pool.len - k);
                    std.mem.swap(u32, &pool[k], &pool[j]);
                    table[r * TOPK + k] = pool[k];
                    if (!seen[pool[k]]) {
                        seen[pool[k]] = true;
                        distinct += 1;
                    }
                }
            }
            distinct_sum += @floatFromInt(distinct);
            const tshape = [_]c_int{ S, TOPK };
            a.* = mlx.mlx_array_new_data(&table, &tshape, 2, .uint32);
            made += 1;
        }
        in.distinct_avg = distinct_sum / @as(f64, CALLS_SMALL);
        try evalAll(&.{ in.x, in.x_act, in.scores });
        {
            const a = try rowsCall(&in, .pair, 0);
            defer _ = mlx.mlx_array_free(a);
            try std.testing.expect(try allFinite(a, s));
        }
        const gu_bytes = in.distinct_avg * gu_per_expert;
        const dn_bytes = in.distinct_avg * dn_per_expert;
        try benchRows(&in, .gateup, "rows gate+up", gu_bytes);
        try benchRows(&in, .downred, "rows down+reduce", dn_bytes);
        try benchRows(&in, .pair, "rows pair", gu_bytes + dn_bytes);
        try benchChain(&in, .pair, "rows pair CHAIN", gu_bytes + dn_bytes);
        try benchChain(&in, .gateup, "gate+up CHAIN", gu_bytes);
        try benchChain(&in, .downred, "down CHAIN", dn_bytes);
        try benchChain(&in, .ops, "dependency ops CHAIN", 1.0);
        try benchRows(&in, .decode_pair_x_rows, "S decode pairs", gu_bytes + dn_bytes);
    }
    try std.testing.expect(!mlx.errorPending());
}

// A streaming-read reference for the GB/s above: sum over a 2 GiB bf16 array reads every byte once, far
// larger than any cache, so its rate is the DRAM read rate this harness can reach.
// The read ceiling with wide loads: one simdgroup streams its own 64 KiB span with 16-byte loads, UNROLL
// loads in flight per lane, and folds what it read into one word so nothing is dead code. A DRAM rate here
// that beats the reduction above says the reduction was the limit, not the memory.
const STREAM_WIDE_SOURCE =
    \\uint sg = thread_position_in_grid.x / 32;
    \\auto lane = thread_index_in_simdgroup;
    \\const device uint4* p = (const device uint4*)a + (size_t)sg * (size_t)SPAN4;
    \\uint acc = 0;
    \\if (STRIDED) {
    \\  // grid-stride: at each step the whole GPU reads one contiguous slab
    \\  const device uint4* q = (const device uint4*)a;
    \\  size_t gid = thread_position_in_grid.x;
    \\  size_t nth = (size_t)NSG * 32;
    \\  for (size_t i = gid; i < nth * (size_t)SPAN4 / 32 ; i += nth * (size_t)UNROLL) {
    \\    uint4 v[UNROLL];
    \\    for (int u = 0; u < UNROLL; ++u) v[u] = q[i + nth * (size_t)u];
    \\    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    \\  }
    \\  out[sg] = simd_sum(acc);
    \\  return;
    \\}
    \\for (size_t i = lane; i < (size_t)SPAN4; i += (size_t)(32 * UNROLL)) {
    \\  uint4 v[UNROLL];
    \\  for (int u = 0; u < UNROLL; ++u) v[u] = p[i + (size_t)(32 * u)];
    \\  for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    \\}
    \\out[sg] = simd_sum(acc);
;

test "stream read peak: wide loads over 2 GiB (MOE_UBENCH=1)" {
    if (std.c.getenv("MOE_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x30E0B6);
    const rnd = prng.random();
    const rows: c_int = 65536;
    const cols: c_int = 16384;
    const a = try randUniformBf16(rnd, &.{ rows, cols }, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_array_eval(a));
    const bytes = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) * 2.0;
    const span4: c_int = 4096; // uint4 per simdgroup: 64 KiB
    const nsg: c_int = @intCast(@divExact(@as(i64, rows) * cols * 2, @as(i64, span4) * 16));
    const io = std.Io.Threaded.global_single_threaded.io();
    const ins = [_][*:0]const u8{"a"};
    const outs = [_][*:0]const u8{"out"};
    const vin = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(vin);
    const vout = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(vout);
    const kernel = mlx.mlx_fast_metal_kernel_new("stream_wide", vin, vout, STREAM_WIDE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    for ([_]c_int{ 1, 2, 4, 8, 11, 12, 14 }) |unroll_code| {
        const strided: c_int = if (unroll_code > 10) 1 else 0;
        const unroll: c_int = if (unroll_code > 10) unroll_code - 10 else unroll_code;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const oshape = [_]c_int{nsg};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &oshape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nsg * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPAN4", span4));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "UNROLL", unroll));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "STRIDED", strided));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", nsg));
        const in_arr = [_]mlx.mlx_array{a};
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, in_arr.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var times: [BATCHES]u64 = undefined;
        for (0..WARM + BATCHES) |b| {
            var sw = io_util.Stopwatch.init(io);
            var out_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(out_vec);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, s));
            var o = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_vector_array_get(&o, out_vec, 0));
            try mlx.check(mlx.mlx_array_eval(o));
            if (b >= WARM) times[b - WARM] = sw.read();
        }
        std.mem.sort(u64, &times, {}, std.sort.asc(u64));
        const med = @as(f64, @floatFromInt(times[BATCHES / 2]));
        const best = @as(f64, @floatFromInt(times[0]));
        std.debug.print("[moe-rows-ubench] stream wide-load unroll={d} strided={d}: median {d:.2} ms = {d:.0} GB/s, best {d:.2} ms = {d:.0} GB/s\n", .{ unroll, strided, med / 1e6, bytes / med, best / 1e6, bytes / best });
    }
    try std.testing.expect(!mlx.errorPending());
}

test "stream read peak: reduction over 2 GiB (MOE_UBENCH=1)" {
    if (std.c.getenv("MOE_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x30E0B5);
    const rnd = prng.random();
    const rows: c_int = 65536;
    const cols: c_int = 16384;
    const a = try randUniformBf16(rnd, &.{ rows, cols }, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_array_eval(a));
    const bytes = @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(cols)) * 2.0;
    const io = std.Io.Threaded.global_single_threaded.io();
    var times: [BATCHES]u64 = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_sum_axis(&out, a, 1, false, s));
        try mlx.check(mlx.mlx_array_eval(out));
        if (b >= WARM) times[b - WARM] = sw.read();
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const med = @as(f64, @floatFromInt(times[BATCHES / 2]));
    const best = @as(f64, @floatFromInt(times[0]));
    std.debug.print("[moe-rows-ubench] stream read peak: sum over {d:.2} GiB bf16: median {d:.2} ms = {d:.0} GB/s, best {d:.2} ms = {d:.0} GB/s\n", .{
        bytes / 1073741824.0, med / 1e6, bytes / med, best / 1e6, bytes / best,
    });
    try std.testing.expect(!mlx.errorPending());
}
