//! Exact physical grid transpose for GLM routed prefill.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const base = @import("expert_exl3_kernels.zig");
const log = @import("mlx_host").log;
const api = @import("root.zig");
const support = base.PrefillGridSupport;
const Arr = mlx.mlx_array;
var calls: usize = 0;
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}
fn replace(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    const at = comptime std.mem.indexOf(u8, source, old).?;
    return source[0..at] ++ value ++ source[at + old.len ..] ++ "";
}
const SOURCE = replace(replace(support.source, "uint win = uint(threadgroup_position_in_grid.y);", "uint win = uint(threadgroup_position_in_grid.x);"), "uint(threadgroup_position_in_grid.x) * 128u + sg * 32u", "uint(threadgroup_position_in_grid.y) * 128u + sg * 32u");
var kernels: [base.SLOT_ROWS][api.format.Window.count]?mlx.mlx_fast_metal_kernel = @splat(@splat(null));
var failed: [base.SLOT_ROWS][api.format.Window.count]bool = @splat(@splat(false));
var build_attempts: usize = 0;
var force_build_failure = false;
fn build(dec: api.format.Decode) !mlx.mlx_fast_metal_kernel {
    build_attempts += 1;
    if (force_build_failure) return error.MetalKernelCompileFailed;
    return support.makeKernel(SOURCE, dec);
}
const Config = struct { input: c_int, output: c_int, rate: c_int, value: mlx.mlx_fast_metal_kernel_config };
var configs: [8]?Config = @splat(null);
fn project(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, x: Arr, bank: Arr, ids: Arr, starts: Arr, live: Arr, windows: c_int) !Arr {
    const k = mlx.getShape(x)[1];
    const n = mlx.getShape(bank)[2] * 16;
    const rate = mlx.getShape(bank)[3];
    var cached: ?mlx.mlx_fast_metal_kernel_config = null;
    for (configs) |entry| if (entry) |e| if (e.input == k and e.output == n and e.rate == rate) {
        cached = e.value;
    };
    var owned = false;
    const cfg = cached orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 16384, n }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        inline for (.{ "IDIM", "ODIM", "NHW", "WIN" }, .{ k, n, rate, @as(c_int, 32) }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, value));
        for (&configs) |*entry| if (entry.* == null) {
            entry.* = .{ .input = k, .output = n, .rate = rate, .value = c };
            break :blk c;
        };
        owned = true;
        break :blk c;
    };
    defer if (owned) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * windows, @divExact(n, 128), 1));
    const iv = mlx.mlx_vector_array_new_data(&.{ x, bank, ids, starts, live }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var output = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_vector_array_get(&output, ov, 0));
    return output;
}
fn eligible(s: mlx.mlx_stream, x: Arr, bank: api.Bank, indices: Arr, scores: Arr) bool {
    for ([_]Arr{ x, indices, scores, bank.gate.trellis, bank.gate.suh, bank.gate.svh, bank.up.trellis, bank.up.suh, bank.up.svh, bank.down.trellis, bank.down.suh, bank.down.svh }) |a| if (a.ctx == null) return false;
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 2048, 4096 }) or !std.mem.eql(c_int, mlx.getShape(indices), &.{ 1, 2048, 8 }) or !std.mem.eql(c_int, mlx.getShape(scores), &.{ 1, 2048, 8 }) or mlx.mlx_array_dtype(indices) != .uint32 or mlx.mlx_array_dtype(scores) != .float32) return false;
    inline for (.{ .{ "gate", 4096, 2048 }, .{ "up", 4096, 2048 }, .{ "down", 2048, 4096 } }) |entry| {
        const p = @field(bank, entry[0]);
        api.validateClampedProjection(p, 288, entry[1], entry[2]) catch return false;
    }
    return true;
}
pub fn tryMoe(s: mlx.mlx_stream, x: Arr, bank: api.Bank, indices: Arr, scores: Arr, dec: api.format.Decode, limit: c_int) !?Arr {
    if (!dec.codebook.servesGlmFastPaths() or limit != 10 or !eligible(s, x, bank, indices, scores)) return null;
    if (base.gemmWindowRows() != 32 or !base.gemmWindowAligned()) return null;
    base.setDecodeParams(dec);
    if (!support.available()) return null;
    const result = try moe(s, x, bank, indices, scores, dec);
    if (result != null) calls += 1;
    return result;
}
pub fn moe(s: mlx.mlx_stream, x: Arr, bank: api.Bank, indices: Arr, scores: Arr, dec: api.format.Decode) !?Arr {
    if (!dec.codebook.servesGlmFastPaths() or !eligible(s, x, bank, indices, scores)) return null;
    base.setDecodeParams(dec);
    const c = base.slotRow(dec.codebook, false);
    const w = dec.window.index();
    // A probe eval drains the GPU, so a failed kernel is declined for good.
    if (failed[c][w]) return null;
    const slot = &kernels[c][w];
    if (slot.* == null) slot.* = build(dec) catch {
        failed[c][w] = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] GLM grid transpose declined (kernel probe failed); the clamped sorted GEMM serves prefill\n", .{});
        return null;
    };
    const kernel = slot.*.?;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ 2048, 4096 }, 2, s));
    try mlx.check(mlx.mlx_reshape(&ids, indices, &.{16384}, 1, s));
    try mlx.check(mlx.mlx_reshape(&sc, scores, &.{16384}, 1, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, ids, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, ids, order, 0, s));
    const prep = try support.prepare(s, flat, bank.gate.suh, bank.up.suh, sorted, order_i, 4096, 16384, 8);
    defer for ([_]Arr{ prep[0], prep[1] }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const metadata = try support.windows(s, sorted, order_i, 16384, 32, 288, true);
    defer for ([_]Arr{ metadata.inverse, metadata.table.starts, metadata.table.nlives }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (metadata.inverse.ctx == null) return error.ExpectedGlmInverseRouting;
    const tab = metadata.table;
    const gate = try project(s, kernel, prep[0], bank.gate.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(gate);
    const up = try project(s, kernel, prep[1], bank.up.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(up);
    const middle = try support.middle(s, gate, up, bank.gate.svh, bank.up.svh, bank.down.suh, sorted, 2048, 16384, 10);
    defer _ = mlx.mlx_array_free(middle);
    const down = try project(s, kernel, middle, bank.down.trellis, sorted, tab.starts, tab.nlives, tab.nwin);
    defer _ = mlx.mlx_array_free(down);
    const result = try support.finish(s, down, metadata.inverse, bank.down.svh, ids, sc, 4096, 2048, 8, .bfloat16);
    defer _ = mlx.mlx_array_free(result);
    var shaped = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, result, &.{ 1, 2048, 4096 }, 3, s));
    return shaped;
}

test "GLM prefill grid declines guards before dispatch" {
    const count = dispatchCount();
    const nil = Arr{ .ctx = null };
    const p = api.Proj{ .trellis = nil, .suh = nil, .svh = nil };
    const bank = api.Bank{ .gate = p, .up = p, .down = p };
    try std.testing.expect((try tryMoe(mlx.gpuStream(), nil, bank, nil, nil, .{ .codebook = .mul1, .window = .w12 }, 10)) == null);
    try std.testing.expect((try tryMoe(mlx.gpuStream(), nil, bank, nil, nil, .{ .codebook = .mcg, .window = .w12 }, 9)) == null);
    try std.testing.expect((try tryMoe(mlx.gpuStream(), nil, bank, nil, nil, .{ .codebook = .mcg, .window = .w12 }, 10)) == null);
    try std.testing.expectEqual(count, dispatchCount());
}

const Owned = struct {
    list: std.ArrayList(*Arr) = .empty,
    fn deinit(self: *Owned) void {
        for (self.list.items) |a| {
            _ = mlx.mlx_array_free(a.*);
            std.testing.allocator.destroy(a);
        }
        self.list.deinit(std.testing.allocator);
    }
    fn slot(self: *Owned) !*Arr {
        const a = try std.testing.allocator.create(Arr);
        a.* = mlx.mlx_array_new();
        self.list.append(std.testing.allocator, a) catch |e| {
            std.testing.allocator.destroy(a);
            return e;
        };
        return a;
    }
    fn own(self: *Owned, value: Arr) !*Arr {
        const a = try self.slot();
        _ = mlx.mlx_array_free(a.*);
        a.* = value;
        return a;
    }
};

fn testProj(o: *Owned, s: mlx.mlx_stream, input: c_int, output: c_int, rate: c_int, seed: u32) !api.Proj {
    const key = try o.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const trellis = try o.slot();
    try mlx.check(mlx.mlx_random_bits(trellis, &[_]c_int{ 288, @divExact(input, 16), @divExact(output, 16), rate }, 4, 2, key.*, s));
    const lo = try o.slot();
    try mlx.check(mlx.mlx_astype(lo, (try o.own(mlx.mlx_array_new_float(-0.2))).*, .float16, s));
    const hi = try o.slot();
    try mlx.check(mlx.mlx_astype(hi, (try o.own(mlx.mlx_array_new_float(0.2))).*, .float16, s));
    const suh = try o.slot();
    try mlx.check(mlx.mlx_random_uniform(suh, lo.*, hi.*, &[_]c_int{ 288, input }, 2, .float16, key.*, s));
    const svh = try o.slot();
    try mlx.check(mlx.mlx_random_uniform(svh, lo.*, hi.*, &[_]c_int{ 288, output }, 2, .float16, key.*, s));
    return .{ .trellis = trellis.*, .suh = suh.*, .svh = svh.* };
}

/// The grid chain must return the sorted chain's bytes; `skewed` piles half the tokens on eight experts.
fn gridMatchesSortedChain(rates: [3]c_int, skewed: bool, dec: api.format.Decode) !void {
    const s = mlx.gpuStream();
    if (!support.available()) return error.SkipZigTest;
    // Each rate allocates several GB of trellis; return it before the next case.
    defer _ = mlx.mlx_clear_cache();
    var o: Owned = .{};
    defer o.deinit();
    const bank = api.Bank{
        .gate = try testProj(&o, s, 4096, 2048, rates[0], 11),
        .up = try testProj(&o, s, 4096, 2048, rates[1], 13),
        .down = try testProj(&o, s, 2048, 4096, rates[2], 17),
    };
    const key = try o.slot();
    try mlx.check(mlx.mlx_random_key(key, 19));
    const x32 = try o.slot();
    try mlx.check(mlx.mlx_random_normal(x32, &[_]c_int{ 1, 2048, 4096 }, 3, .float32, 0, 1, key.*, s));
    const x = try o.slot();
    try mlx.check(mlx.mlx_astype(x, x32.*, .bfloat16, s));
    var id_data: [2048 * 8]u32 = undefined;
    var score_data: [2048 * 8]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (0..2048) |t| {
        const span: u32 = if (skewed and t % 2 == 0) 8 else 288;
        var taken: [8]u32 = undefined;
        for (0..8) |j| {
            var e = rnd.uintLessThan(u32, span);
            while (std.mem.indexOfScalar(u32, taken[0..j], e) != null) e = (e + 1) % span;
            taken[j] = e;
            id_data[t * 8 + j] = e;
            score_data[t * 8 + j] = rnd.float(f32) + 0.1;
        }
    }
    const ids = try o.own(mlx.mlx_array_new_data(&id_data, &[_]c_int{ 1, 2048, 8 }, 3, .uint32));
    const scores = try o.own(mlx.mlx_array_new_data(&score_data, &[_]c_int{ 1, 2048, 8 }, 3, .float32));
    // A hash-mask codebook's reference is its sorted chain with the scale decoded into every weight.
    base.unfolded_for_test = dec.codebook.tabledHashMask();
    defer base.unfolded_for_test = false;
    const want = try o.own(try api.moeClamped(s, x.*, bank, ids.*, scores.*, dec, 10));
    try mlx.check(mlx.mlx_array_eval(want.*));
    base.unfolded_for_test = false;
    const before = dispatchCount();
    const got = try o.own((try tryMoe(s, x.*, bank, ids.*, scores.*, dec, 10)) orelse return error.TestExpectedGlmGrid);
    try std.testing.expectEqual(before + 1, dispatchCount());
    try mlx.check(mlx.mlx_array_eval(want.*));
    try mlx.check(mlx.mlx_array_eval(got.*));
    try std.testing.expectEqual(mlx.mlx_array_dtype(want.*), mlx.mlx_array_dtype(got.*));
    const count = mlx.mlx_array_size(want.*);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want.*).?[0..count], mlx.mlx_array_data_bfloat16(got.*).?[0..count]);
}

const MCG_W12: api.format.Decode = .{ .codebook = .mcg, .window = .w12 };

test "GLM prefill grid serves every admitted rate, mixed projections included, with the sorted chain's bytes" {
    const expert = @import("expert_exl3.zig");
    var n: c_int = expert.Rate.min_n;
    while (n <= expert.Rate.max_n) : (n += 2) {
        try gridMatchesSortedChain(.{ n, n, n }, @mod(n, 8) == 0, MCG_W12);
    }
    try gridMatchesSortedChain(.{ 40, 40, 36 }, true, MCG_W12);
    try gridMatchesSortedChain(.{ 36, 40, 48 }, false, MCG_W12);
    try gridMatchesSortedChain(.{ 32, 64, 40 }, true, MCG_W12);
}

test "GLM prefill grid serves W14 packs with the sorted chain's bytes" {
    try gridMatchesSortedChain(.{ 36, 36, 36 }, true, .{ .codebook = .mcg, .window = .w14 });
    try gridMatchesSortedChain(.{ 40, 40, 40 }, false, .{ .codebook = .mcg, .window = .w14 });
    try gridMatchesSortedChain(.{ 40, 40, 36 }, true, .{ .codebook = .mcg, .window = .w14 });
}

test "GLM serves spec1 and std1 on the prefill grid with the unfolded sorted chain's bytes" {
    try gridMatchesSortedChain(.{ 36, 40, 48 }, true, api.format.Decode.tabled(.spec1, .w15));
    try gridMatchesSortedChain(.{ 36, 40, 48 }, true, api.format.Decode.tabled(.std1, .w15));
    // A per-rate SPEC1 table at a window other than its own: the three projections decode rows of three rates.
    const table = api.codebook_tables.indexOf("spec1-test-per-rate");
    try gridMatchesSortedChain(.{ 24, 32, 48 }, true, .{ .codebook = .spec1, .window = .w9, .table = table });
    try gridMatchesSortedChain(.{ 40, 42, 64 }, false, .{ .codebook = .spec1, .window = .w16, .table = table });
}

test "GLM serves PAIR1 on the prefill grid with the unfolded sorted chain's bytes" {
    try gridMatchesSortedChain(.{ 36, 36, 36 }, true, .{ .codebook = .pair1, .window = .w16 });
    try gridMatchesSortedChain(.{ 24, 32, 48 }, true, .{ .codebook = .pair1, .window = .w16 });
    try gridMatchesSortedChain(.{ 40, 42, 64 }, false, .{ .codebook = .pair1, .window = .w16 });
}

test "GLM serves std1 on the prefill grid and probes a failing kernel once per codebook and window" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const saved_kernels = kernels;
    kernels = @splat(@splat(null));
    build_attempts = 0;
    force_build_failure = true;
    defer {
        kernels = saved_kernels;
        force_build_failure = false;
        build_attempts = 0;
        failed = @splat(@splat(false));
    }
    failed = @splat(@splat(false));
    var o: Owned = .{};
    defer o.deinit();
    const bank = api.Bank{
        .gate = try testProj(&o, s, 4096, 2048, 32, 11),
        .up = try testProj(&o, s, 4096, 2048, 32, 13),
        .down = try testProj(&o, s, 2048, 4096, 32, 17),
    };
    const x = try o.own(mlx.mlx_array_new());
    try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ 1, 2048, 4096 }, 3, .bfloat16, s));
    const ids = try o.own(mlx.mlx_array_new());
    try mlx.check(mlx.mlx_zeros(ids, &[_]c_int{ 1, 2048, 8 }, 3, .uint32, s));
    const scores = try o.own(mlx.mlx_array_new());
    try mlx.check(mlx.mlx_zeros(scores, &[_]c_int{ 1, 2048, 8 }, 3, .float32, s));
    const dec = api.format.Decode{ .codebook = .mcg, .window = .w12 };
    try std.testing.expect((try moe(s, x.*, bank, ids.*, scores.*, dec)) == null);
    try std.testing.expect((try moe(s, x.*, bank, ids.*, scores.*, dec)) == null);
    try std.testing.expectEqual(@as(usize, 1), build_attempts);
    try std.testing.expect((try moe(s, x.*, bank, ids.*, scores.*, .{ .codebook = .mcg, .window = .w14 })) == null);
    try std.testing.expectEqual(@as(usize, 2), build_attempts);
    for ([_]api.format.Window{ .w15, .w12 }, [_]usize{ 3, 4 }) |window, attempts| for (0..2) |_| {
        try std.testing.expect((try moe(s, x.*, bank, ids.*, scores.*, .{ .codebook = .std1, .window = window, .table = api.codebook_tables.indexOf("std1") })) == null);
        try std.testing.expectEqual(attempts, build_attempts);
    };
}
