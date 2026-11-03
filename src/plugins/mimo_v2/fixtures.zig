//! plugins/mimo_v2/fixtures: the served MiMo (this plugin behind the driver) against modeling_mimo_v2.py's logits on a
//! tiny random MiMo stored like the release (`tests/dump_mimo_v2_fixtures.py`, packed by `tests/convert_mimo_v2.py`).
//! Owns tests only; model-gated (`MIMO_V2_MODEL`, `MIMO_V2_FIXTURE`, `MIMO_V2_MTP_FIXTURE`).

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
const env = @import("../../env.zig");
const plugin_head = @import("../../engines/mlx/plugin/head.zig");
const MtpHeadRef = @import("../../engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;

// ── tests ──

test "mimo_v2 batched decode: a slot in a two-slot tick tracks the fp32 reference as well as serial decode does (MIMO_V2_MODEL, MIMO_V2_FIXTURE)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_FIXTURE") orelse return error.SkipZigTest;
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
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..T];
    const v: usize = @intCast(config.vocab_size);
    const ref = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref);

    // A MiMo slot owns a KV cache and a position, no recurrent state.
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
    // Prompts of different length, decoded past the 8-token window. Slot A
    // decodes the fixture's own sequence, so every step has a reference row.
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
        const out = try xfm.forwardBatchedDecode(&.{ @intCast(ta), @intCast(tb) }, &ctxs, &offs);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        // The scheduler advances each slot's own position after the tick.
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
    }
    std.debug.print("[mimo batched] {d} ticks: min cos vs fp32, serial {d:.5} batched {d:.5}; batched decided misses {d}\n", .{ K, cos_serial, cos_batched, misses });
    try testing.expect(cos_batched > cos_serial - 0.001 and misses == 0);
}

test "mimo_v2 fixture: one-shot and chunked prefill + decode vs modeling_mimo_v2.py (MIMO_V2_MODEL, MIMO_V2_FIXTURE)" {
    // A tiny random MiMo stored like the release (tests/dump_mimo_v2_fixtures.py), packed by
    // tests/convert_mimo_v2.py --bits 16; logits from the checkpoint's own code in f32.
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_FIXTURE") orelse return error.SkipZigTest;
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

    // [a] the whole prompt in one forward: every row, sliding window included.
    const full = try Run.rows(&xfm, ids_arr, 0, T, s);
    defer _ = mlx.mlx_array_free(full);
    const ours_full = try qwen4ReadF32(allocator, full, s);
    defer allocator.free(ours_full);
    const a = qwen4CompareRows(ours_full, ref, @intCast(T), v);
    std.debug.print("[mimo fixture] one-shot {d} rows: min cos {d:.5}, argmax {d}/{d}\n", .{ a.rows, a.min_cos, a.argmax_agree, a.rows });
    try testing.expect(a.min_cos > 0.995 and mimoDecidedMisses(ours_full, ref, @intCast(T), v) == 0);

    // [b] a chunked prefill split across the window, then single-token decode steps.
    try xfm.resetCache();
    const t_pre: c_int = T - 6;
    const chunks = [_][2]c_int{ .{ 0, 13 }, .{ 13, t_pre } };
    for (chunks) |c| {
        const out = try Run.rows(&xfm, ids_arr, c[0], c[1], s);
        _ = mlx.mlx_array_free(out);
    }
    var worst: f64 = 1.0;
    var agree: usize = 0;
    var t = t_pre;
    while (t < T) : (t += 1) {
        const out = try Run.rows(&xfm, ids_arr, t, t + 1, s);
        defer _ = mlx.mlx_array_free(out);
        const row = try qwen4ReadF32(allocator, out, s);
        defer allocator.free(row);
        const r = qwen4CompareRows(row, ref[@as(usize, @intCast(t)) * v ..], 1, v);
        worst = @min(worst, r.min_cos);
        agree += r.argmax_agree;
    }
    std.debug.print("[mimo fixture] chunked prefill + {d} decode steps: min cos {d:.5}, argmax {d}/{d}\n", .{ T - t_pre, worst, agree, T - t_pre });
    try testing.expect(worst > 0.995 and agree == @as(usize, @intCast(T - t_pre)));
}

fn readF32(a: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    return a.dupe(f32, src[0..n]);
}

/// Rows `[from, to)` of a `[T, D]` fixture tensor as `[1, to - from, D]`.
fn fixtureRows(fx: *const model_mod.Weights, key: []const u8, from: usize, to: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const arr = fx.get(key) orelse return error.MissingFixtureTensor;
    const sh = mlx.getShape(arr);
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_slice(&part, arr, &.{ @intCast(from), 0 }, 2, &.{ @intCast(to), sh[1] }, 2, &.{ 1, 1 }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, part, &.{ 1, @intCast(to - from), sh[1] }, 3, s));
    return out;
}

/// A bf16 pack against the f32 rendering: the worst error relative to the row's peak stays under `bar`.
fn expectRowClose(label: []const u8, k: usize, q: usize, ours: mlx.mlx_array, reference: mlx.mlx_array, bar: f32, s: mlx.mlx_stream) !void {
    const a = try readF32(std.testing.allocator, ours, s);
    defer std.testing.allocator.free(a);
    const b = try readF32(std.testing.allocator, reference, s);
    defer std.testing.allocator.free(b);
    try std.testing.expectEqual(b.len, a.len);
    var worst: f32 = 0;
    var peak: f32 = 0;
    for (a, b) |x, y| {
        try std.testing.expect(std.math.isFinite(x));
        worst = @max(worst, @abs(x - y));
        peak = @max(peak, @abs(y));
    }
    if (worst > bar * peak) {
        std.debug.print("mimo mtp {s}: head {d} row {d} max |d| {d} > {d} of peak {d}\n", .{ label, k, q, worst, bar, peak });
        return error.TestExpectedEqual;
    }
}

test "mimo mtp heads track the torch rendering of the MiMo-V2 MTP layer across rounds, drafts and rollbacks (MIMO_V2_MODEL, MIMO_V2_MTP_FIXTURE)" {
    // Head k's row p pairs the trunk hidden at p with token x_{p+k+1} (tests/dump_mimo_v2_fixtures.py).
    const model_dir = env.get(.MIMO_V2_MODEL) orelse return error.SkipZigTest;
    const fixture_path = env.get(.MIMO_V2_MTP_FIXTURE) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer config.deinit(a);
    var xfm = try driver.load(io, a, &config, std.mem.span(model_dir), s);
    defer xfm.deinit();
    const served = plugin_head.load(io, a, &xfm, std.mem.span(model_dir)) orelse return error.NoMtpHeads;
    defer plugin_head.destroy(served, a);
    try std.testing.expectEqual(@as(u32, 3), served.heads);
    const h: MtpHeadRef = .{ .plugin = served };
    var cr = try h.makeCache(a);
    defer cr.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const t_total = mlx.mlx_array_size(ids_arr);
    const ids_src = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..t_total];
    const x = try a.alloc(u32, t_total);
    defer a.free(x);
    for (x, ids_src) |*v, id| v.* = @intCast(id);

    // History in uneven chunks past the window (8): rows p in [0, 30) with x_{p+1}.
    var p: usize = 0;
    for ([_]usize{ 13, 11, 6 }) |n| {
        const hid = try fixtureRows(&fx, "target_hidden", p, p + n, s);
        defer _ = mlx.mlx_array_free(hid);
        try h.appendHistory(&xfm, &cr, x[p + 1 .. p + n + 1], hid, @intCast(p), null, a);
        p += n;
    }

    // Rounds as the generator runs them: each rolls back to the previous round's q and opens on the committed rows
    // from there to its own q; a wrong round drafts tokens the reference never saw.
    const plan = [_]struct { wrong: bool, accept: usize }{
        .{ .wrong = false, .accept = 3 }, .{ .wrong = true, .accept = 0 }, .{ .wrong = false, .accept = 1 },
        .{ .wrong = false, .accept = 2 }, .{ .wrong = true, .accept = 0 }, .{ .wrong = false, .accept = 3 },
    };
    var stash_from: usize = p;
    var q = p;
    var rounds: usize = 0;
    for (plan) |round| {
        if (q + 4 >= t_total) break;
        try cr.truncate(stash_from, s);
        const hid = try fixtureRows(&fx, "target_hidden", stash_from, q + 1, s);
        defer _ = mlx.mlx_array_free(hid);
        const committed = x[stash_from + 1 .. q + 2];
        const open_ids = try a.alloc(i32, committed.len);
        defer a.free(open_ids);
        for (open_ids, committed) |*o, t| o.* = @intCast(t);
        const open_arr = mlx.mlx_array_new_data(open_ids.ptr, &[_]c_int{@intCast(open_ids.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(open_arr);
        const o0 = try h.forward(&xfm, &cr, open_arr, hid, @intCast(stash_from), .mixed, null, committed);
        defer _ = mlx.mlx_array_free(o0.rerank_x);
        var trunk = o0.hidden_next;
        defer _ = mlx.mlx_array_free(trunk);
        {
            const ref = try fixtureRows(&fx, "mtp0_out", q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref);
            try expectRowClose("out", 0, q, o0.rerank_x, ref, 0.02, s);
            const logits = try xfm.lmHeadForeign(o0.rerank_x);
            defer _ = mlx.mlx_array_free(logits);
            const ref_l = try fixtureRows(&fx, "mtp0_logits", q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref_l);
            try expectRowClose("logits", 0, q, logits, ref_l, 0.02, s);
        }
        for (1..3) |i| {
            const truth = x[q + 1 + i];
            const tok: i32 = @intCast(if (round.wrong) (truth + 7) % @as(u32, @intCast(config.vocab_size)) else truth);
            const d = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32);
            defer _ = mlx.mlx_array_free(d);
            const out = try h.forward(&xfm, &cr, d, trunk, @intCast(q + i), .mixed, null, null);
            defer _ = mlx.mlx_array_free(out.rerank_x);
            _ = mlx.mlx_array_free(trunk);
            trunk = out.hidden_next;
            if (round.wrong) continue;
            var key: [16]u8 = undefined;
            const ref = try fixtureRows(&fx, try std.fmt.bufPrint(&key, "mtp{d}_out", .{i}), q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref);
            try expectRowClose("out", i, q, out.rerank_x, ref, 0.02, s);
        }
        stash_from = q;
        q += round.accept + 1;
        rounds += 1;
    }
    try std.testing.expect(rounds >= 5);
}
