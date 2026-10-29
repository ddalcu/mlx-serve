//! `format.zig`'s tests, out of its import graph: a plugin unit that reaches `format.zig` never compiles them.

const format = @import("format.zig");
const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const native = @import("../native.zig");
const rht = @import("rht.zig");
const Verdict = abi.Verdict;
const arr = native.arr;
const wrap = native.wrap;
const NEG_SCALE = format.NEG_SCALE;
const table = format.table;
const transformer = @import("../../../engines/mlx/transformer.zig");
const testing = std.testing;
const ops = @import("../../../mlx/ops.zig");
const parity = @import("../../testing/parity.zig");
const resident = @import("../../weights/resident.zig");
const registerRhtSigns = @import("../hadamard.zig").registerRhtSigns;

fn f16Rand(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const b = try parity.testRandWeightBf16(rnd, shape, s);
    defer _ = mlx.mlx_array_free(b);
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&h, b, .float16, s));
    return h;
}

fn signsOf(rnd: std.Random, k: usize) !mlx.mlx_array {
    const v = try testing.allocator.alloc(f16, k);
    defer testing.allocator.free(v);
    for (v) |*e| e.* = if (rnd.boolean()) 1 else -1;
    return mlx.mlx_array_new_data(v.ptr, &[_]c_int{@intCast(k)}, 1, .float16);
}

// Bar: a Prism pack (f16, every shipped rotation block, 2- and 4-bit, biases at -scales or not) loads through the
// format and multiplies bit-equal to the engine's rotated arm (`Transformer.qmatmul` with the pack's registry).
test "quant hadamard: rotated weights multiply bit-equal to the engine's rotated arm" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const N: c_int = 256;
    const K: c_int = 4096;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for ([_]u32{ 512, 1024, 2048, 4096 }) |blk| for ([_]u8{ 2, 4 }) |bits| for ([_]bool{ true, false }) |neg| {
        var weights = resident.Weights.init(testing.allocator);
        const raw = try f16Rand(rnd, &.{ N, K }, s);
        const t = try native.quantized(raw, "affine", bits, 128, s);
        _ = mlx.mlx_array_free(raw);
        var bi = t.bi;
        if (neg) {
            bi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_negative(&bi, t.sc, s));
            _ = mlx.mlx_array_free(t.bi);
        }
        try native.putTriple(&weights, "l.proj", .{ .w = t.w, .sc = t.sc, .bi = bi });
        try weights.put("l.proj.signs", try signsOf(rnd, K));
        const raw_plain = try f16Rand(rnd, &.{ N, K }, s);
        defer _ = mlx.mlx_array_free(raw_plain);
        try native.putTriple(&weights, "l.plain", try native.quantized(raw_plain, "affine", bits, 128, s));
        var r = resident.Resident.adopt(weights);
        defer r.deinit();

        var reg = rht.Registry.init(testing.allocator, blk);
        defer reg.deinit();
        try registerRhtSigns(&reg, &r.weights, s);
        var xfm: transformer.Transformer = undefined;
        xfm.s = s;
        xfm.rht = &reg;
        xfm.ternary_2bit = false;
        xfm.int8_cache = null;
        xfm.bits_cache = .{};
        xfm.config = .{};
        xfm.config.quant_bits = bits;
        xfm.config.quant_group_size = 128;

        var cfg_buf: [96]u8 = undefined;
        const config = try std.fmt.bufPrint(&cfg_buf, "{{\"modules\":[{{\"path\":\"l.proj\",\"block\":{d}}}]}}", .{blk});
        var bj: [64]u8 = undefined;
        const block_json = try std.fmt.bufPrint(&bj, "{{\"bits\":{d},\"group_size\":128}}", .{bits});
        try testing.expectEqual(abi.Claim.by_marker, table.claims(.of(block_json), .of(config)));
        const src = r.source();
        var ctx: abi.LoadCtx = .{ .block_json = .of(block_json), .config_json = .of(config), .source = &src, .stream = .{ .ctx = s.ctx } };
        try testing.expect(table.open(&ctx).isOk());
        defer table.close(&ctx);
        for ([_][]const u8{ "l.proj", "l.plain" }) |base| {
            var w: abi.Weight = .{};
            try testing.expect(table.load_tensor(&ctx, .of(base), @intCast(K), &w).isOk());
            defer table.release(&w);
            const rotated_w = std.mem.eql(u8, base, "l.proj");
            try testing.expectEqual(rotated_w, w.side.ctx != null);
            try testing.expectEqual(rotated_w and neg, w.flags & NEG_SCALE != 0);
            for ([_]c_int{ 1, 3, 9 }) |rows| {
                const x = try f16Rand(rnd, &.{ 1, rows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                var got: abi.Array = .{};
                try testing.expect(table.matmul(&w, wrap(x), .{ .ctx = s.ctx }, &got).isOk());
                defer _ = mlx.mlx_array_free(arr(got));
                reg.dropMemo();
                const want = try xfm.qmatmul(x, native.tensor(&ctx, base, "weight"), native.tensor(&ctx, base, "scales"), native.tensor(&ctx, base, "biases"));
                defer _ = mlx.mlx_array_free(want);
                try ops.expectBitEqual(arr(got), want);
            }
        }
    };
}

test "quant hadamard: expert banks are refused by name" {
    var w: abi.Weight = .{};
    const src: abi.Source = undefined;
    const ctx: abi.LoadCtx = .{ .block_json = .of("{}"), .source = &src };
    try testing.expectEqual(Verdict.Code.unsupported, table.load_bank(&ctx, .of("x"), 0, &w).code);
}
