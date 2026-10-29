//! plugins/flint/fixtures: the served Flint (this plugin behind the driver) against rec_mlx's model. Owns tests only.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const driver = @import("../../engines/mlx/plugin/driver.zig");
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const mimoDecidedMisses = @import("../../sdk/testing/parity.zig").decidedMisses;

// ── tests ──

test "flint fixture: one-shot and chunked prefill + decode vs rec_mlx's model (FLINT_MODEL, FLINT_FIXTURE)" {
    // A tiny random Flint (tests/dump_flint_fixtures.py): logits from Recursive's own MLX model in f32,
    // 40 tokens past the 9-token window. Bar: rows agree and no decided row picks another token.
    const model_dir = std.c.getenv("FLINT_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("FLINT_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer config.deinit(allocator);
    var xfm = try driver.load(io, allocator, &config, std.mem.span(model_dir), s);
    defer xfm.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: c_int = @intCast(mlx.mlx_array_size(ids_arr));
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    const Run = struct {
        fn rows(x: *Transformer, ids: mlx.mlx_array, from: c_int, to: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, ids, &.{from}, 1, &.{to}, 1, &.{1}, 1, st));
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, part, &.{ 1, to - from }, 2, st));
            return x.forward(batched);
        }
    };

    const full = try Run.rows(&xfm, ids_arr, 0, T, s);
    defer _ = mlx.mlx_array_free(full);
    const ours_full = try qwen4ReadF32(allocator, full, s);
    defer allocator.free(ours_full);
    const a = qwen4CompareRows(ours_full, ref, @intCast(T), v);
    std.debug.print("[flint fixture] one-shot {d} rows: min cos {d:.5}, argmax {d}/{d}\n", .{ a.rows, a.min_cos, a.argmax_agree, a.rows });
    try testing.expect(a.min_cos > 0.995 and mimoDecidedMisses(ours_full, ref, @intCast(T), v) == 0);

    try xfm.resetCache();
    const t_pre: c_int = T - 8;
    for ([_][2]c_int{ .{ 0, 13 }, .{ 13, t_pre } }) |c| {
        const out = try Run.rows(&xfm, ids_arr, c[0], c[1], s);
        _ = mlx.mlx_array_free(out);
    }
    var worst: f64 = 1.0;
    var misses: usize = 0;
    var t = t_pre;
    while (t < T) : (t += 1) {
        const out = try Run.rows(&xfm, ids_arr, t, t + 1, s);
        defer _ = mlx.mlx_array_free(out);
        const row = try qwen4ReadF32(allocator, out, s);
        defer allocator.free(row);
        const r_ref = ref[@as(usize, @intCast(t)) * v ..][0..v];
        worst = @min(worst, qwen4CompareRows(row, r_ref, 1, v).min_cos);
        misses += mimoDecidedMisses(row, r_ref, 1, v);
    }
    std.debug.print("[flint fixture] chunked prefill + {d} decode steps: min cos {d:.5}, decided misses {d}\n", .{ T - t_pre, worst, misses });
    try testing.expect(worst > 0.995 and misses == 0);
}

test "flint batched decode: a slot in a two-slot tick tracks rec_mlx's model as well as serial decode does (FLINT_MODEL, FLINT_FIXTURE)" {
    const model_dir = std.c.getenv("FLINT_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("FLINT_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer config.deinit(allocator);
    var xfm = try driver.load(io, allocator, &config, std.mem.span(model_dir), s);
    defer xfm.deinit();
    try testing.expect(xfm.supportsBatchedGdnDecode());

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..T];
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    const Slot = struct {
        cache: KVCache,
        off: usize = 0,
        ctx: ForwardCtx = undefined,
        fn init(sl: *@This(), n_layers: u32) !void {
            const cache = try KVCache.init(testing.allocator, n_layers);
            sl.* = .{ .cache = cache };
            sl.ctx = .{ .cache = &sl.cache, .moe_seq_offset = &sl.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
        }
        fn forward(sl: *@This(), x: *Transformer, toks: []const i32) !mlx.mlx_array {
            const arr = mlx.mlx_array_new_data(toks.ptr, &[_]c_int{ 1, @intCast(toks.len) }, 2, .int32);
            defer _ = mlx.mlx_array_free(arr);
            return x.forwardWith(&sl.ctx, arr);
        }
    };
    // Prompts of different length, both decoded past the fixture's 9-token window.
    const pa: usize = T / 2;
    const pb: usize = pa - 3;
    const K: usize = T - pa;
    const a_ids = ids[0 .. pa + K];
    const b_ids = ids[3 .. 3 + pb + K];
    var slots: [4]Slot = undefined; // serial A, serial B, batched A, batched B
    for (&slots) |*sl| try sl.init(config.num_hidden_layers);
    defer for (&slots) |*sl| sl.cache.deinit();
    for (&slots, [_][]const i32{ a_ids[0..pa], b_ids[0..pb], a_ids[0..pa], b_ids[0..pb] }) |*sl, p| _ = mlx.mlx_array_free(try sl.forward(&xfm, p));

    var cos_serial: f64 = 1.0;
    var cos_batched: f64 = 1.0;
    var misses: usize = 0;
    for (0..K) |k| {
        const ta = a_ids[pa + k];
        const tb = b_ids[pb + k];
        const la = try slots[0].forward(&xfm, &.{ta});
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots[1].forward(&xfm, &.{tb});
        defer _ = mlx.mlx_array_free(lb);
        const ctxs = [_]*ForwardCtx{ &slots[2].ctx, &slots[3].ctx };
        const offs = [_]u32{ @intCast(slots[2].off), @intCast(slots[3].off) };
        try testing.expect(xfm.batchedGdnReady(&ctxs));
        const out = try xfm.forwardMoeBatchedDecode(&.{ @intCast(ta), @intCast(tb) }, &ctxs, &offs, null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        slots[2].off += 1;
        slots[3].off += 1;
        const r = ref[(pa + k) * v ..][0..v];
        const sv = try qwen4ReadF32(allocator, la, s);
        defer allocator.free(sv);
        const bv = try qwen4ReadF32(allocator, out[0], s);
        defer allocator.free(bv);
        cos_serial = @min(cos_serial, qwen4CompareRows(sv, r, 1, v).min_cos);
        cos_batched = @min(cos_batched, qwen4CompareRows(bv, r, 1, v).min_cos);
        misses += mimoDecidedMisses(bv, r, 1, v);
        // Slot B decodes its own shifted sequence: batched must agree with serial B.
        const sb = try qwen4ReadF32(allocator, lb, s);
        defer allocator.free(sb);
        const bb = try qwen4ReadF32(allocator, out[1], s);
        defer allocator.free(bb);
        cos_batched = @min(cos_batched, qwen4CompareRows(bb, sb, 1, v).min_cos);
    }
    std.debug.print("[flint batched] {d} ticks: min cos vs reference, serial {d:.5} batched {d:.5}; batched decided misses {d}\n", .{ K, cos_serial, cos_batched, misses });
    try testing.expect(cos_batched > cos_serial - 0.001 and misses == 0);
}
