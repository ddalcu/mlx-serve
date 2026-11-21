//! The inkling2 table against mlx-lm's Inkling 2 (`tests/dump_inkling2_fixtures.py`): the tiny CPU f32 pack prefilled
//! whole and in chunks, decoded step by step, verified and rolled back; the tiny quantized pack on the GPU against the
//! f32 truth and mlx-lm's own GPU run; an image row refused.

const std = @import("std");
const plugin = @import("plugin.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const parity = @import("../../sdk/testing/parity.zig");

const testing = std.testing;
const VOCAB = 120;

const Tokens = struct { prompt: []const u32, decode: []const u32 };

fn tokens(a: std.mem.Allocator) !std.json.Parsed(Tokens) {
    return std.json.parseFromSlice(Tokens, a, @embedFile("fixtures/tiny/tokens.json"), .{});
}

fn refRows(comptime name: []const u8) []const f32 {
    const bytes = @embedFile("fixtures/" ++ name ++ ".f32");
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// One forward of `toks` from `pos` through the slot's cache; its logits rows as f32.
fn step(a: std.mem.Allocator, p: *conformance.Loaded, cache: *test_cache.TestCache, toks: []const u32, pos: u32, logits: abi.Logits, mode: abi.Mode) ![]f32 {
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const positions = [_]u32{pos};
    var out: abi.ForwardOut = .{};
    const v = p.table.forward(p.model, &.{ .mode = mode, .logits = logits, .rows = @intCast(toks.len), .tokens = .of(toks), .slots = .of(&slot), .positions = .of(&positions), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
    if (!v.isOk()) std.debug.print("[inkling2 proof] forward refused: {s}\n", .{v.why.slice()});
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

fn modeOf(n: usize) abi.Mode {
    return if (n == 1) .decode else .prefill;
}

/// Bar: every row's cosine above 0.999 and no row whose margin decides it picks another token.
fn expectRows(ours: []const f32, ref: []const f32) !void {
    try testing.expectEqual(ref.len, ours.len);
    const rows = ref.len / VOCAB;
    const cmp = parity.qwen4CompareRows(ours, ref, rows, VOCAB);
    const misses = parity.decidedMisses(ours, ref, rows, VOCAB);
    if (cmp.min_cos <= 0.999 or misses > 0) std.debug.print("[inkling2 proof] {d} rows: min cos {d:.6}, decided misses {d}\n", .{ rows, cmp.min_cos, misses });
    try testing.expect(cmp.min_cos > 0.999 and misses == 0);
}

/// copied_from: plugins/deepseek_v41/decode/decode.zig (`relRms`; unchanged)
fn relRms(got: []const f32, want: []const f32) f64 {
    var e: f64 = 0;
    var t: f64 = 0;
    for (got, want) |g, w| {
        const d = @as(f64, g) - @as(f64, w);
        e += d * d;
        t += @as(f64, w) * @as(f64, w);
    }
    return @sqrt(e / @max(t, 1e-300));
}

/// copied_from: plugins/deepseek_v41/decode/decode.zig (`expectNoWorse`; log prefix)
fn expectNoWorse(label: []const u8, new: []const f32, chain: []const f32, truth: []const f32) !void {
    const en = relRms(new, truth);
    const ec = relRms(chain, truth);
    std.debug.print("[inkling2 proof] {s}: ours-truth {e:.2}, mlx-lm-truth {e:.2} (rel rms)\n", .{ label, en, ec });
    for (new) |v| if (!std.math.isFinite(v)) return error.NonFinite;
    if (en > 1.25 * ec + 1e-4) return error.FurtherFromTruth;
}

fn fixtureOf(comptime dir: []const u8) fixture.Fixture {
    return .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile("fixtures/" ++ dir ++ "/config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile("fixtures/" ++ dir ++ "/model.safetensors") },
    } };
}

/// The staged pack's path and directory live in `arena` (the conformance stage never frees them).
fn open(arena: std.mem.Allocator) !conformance.Loaded {
    return conformance.Loaded.open(testing.io, arena, plugin.plugin.name, &plugin.table, fixtureOf("tiny"));
}

test "inkling2 proof: a prefill and four decode steps match mlx-lm's Inkling 2 (CPU f32)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tk = try tokens(a);
    defer tk.deinit();
    var p = try open(arena.allocator());
    defer p.close();
    var cache = try test_cache.TestCache.init(a, p.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const n: u32 = @intCast(tk.value.prompt.len);
    const pre = try step(a, &p, &cache, tk.value.prompt, 0, .all_rows, .prefill);
    defer a.free(pre);
    try expectRows(pre, refRows("tiny/prefill_logits"));
    for (tk.value.decode, 0..) |t, i| {
        const row = try step(a, &p, &cache, &.{t}, n + @as(u32, @intCast(i)), .last_row, .decode);
        defer a.free(row);
        try expectRows(row, refRows("tiny/decode_logits")[i * VOCAB ..][0..VOCAB]);
    }
}

test "inkling2 proof: a prefill in chunks of 5, 7, 9 and 11 rows matches the reference on every row" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tk = try tokens(a);
    defer tk.deinit();
    var p = try open(arena.allocator());
    defer p.close();
    var cache = try test_cache.TestCache.init(a, p.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const ref = refRows("tiny/prefill_logits");
    var at: usize = 0;
    for ([_]usize{ 5, 7, 9, 11 }) |n| {
        const rows = try step(a, &p, &cache, tk.value.prompt[at .. at + n], @intCast(at), .all_rows, .prefill);
        defer a.free(rows);
        try expectRows(rows, ref[at * VOCAB .. (at + n) * VOCAB]);
        at += n;
    }
    try testing.expectEqual(tk.value.prompt.len, at);
}

/// `x` (f32, `[1, rows, C]`) as host values.
fn hostF32(a: std.mem.Allocator, x: mlx.mlx_array) ![]f32 {
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, x));
    return conformance.consumeF32(a, .{ .ctx = h.ctx });
}

test "inkling2 proof: a verify forward's conv capture rolls back to its accepted prefix" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tk = try tokens(a);
    defer tk.deinit();
    var p = try open(arena.allocator());
    defer p.close();
    const visits = p.parsed.shell.num_hidden_layers;
    const prompt = tk.value.prompt;
    const n: u32 = @intCast(prompt.len);
    const draft = tk.value.decode[0..3];
    // One cache verifies the draft; two more, from the same prefill, take its first one and two rows plainly.
    var caches: [3]test_cache.TestCache = undefined;
    for (&caches) |*c| c.* = try test_cache.TestCache.init(a, visits);
    defer for (&caches) |*c| c.deinit();
    for (&caches) |*c| a.free(try step(a, &p, c, prompt, 0, .last_row, .prefill));
    a.free(try step(a, &p, &caches[0], draft, n, .all_rows, .verify));
    for ([_]usize{ 0, 1 }) |acc| {
        const c = &caches[1 + acc];
        a.free(try step(a, &p, c, draft[0 .. 1 + acc], n, .last_row, modeOf(1 + acc)));
        for (0..visits) |v| {
            const cap = caches[0].layers[v].conv_input;
            try testing.expect(cap.ctx != null);
            const sh = mlx.getShape(cap);
            // [1, K-1 + rows, C]: rows [1+acc, 1+acc+K-1) are the state after 1+acc accepted rows.
            const width: usize = @intCast(sh[2]);
            const k1: usize = @as(usize, @intCast(sh[1])) - draft.len;
            const all = try hostF32(a, cap);
            defer a.free(all);
            const want = try hostF32(a, c.layers[v].conv);
            defer a.free(want);
            const got = all[(1 + acc) * width ..][0 .. k1 * width];
            try testing.expectEqual(want.len, got.len);
            var worst: f32 = 0;
            var peak: f32 = 0;
            for (got, want) |x, y| {
                worst = @max(worst, @abs(x - y));
                peak = @max(peak, @abs(y));
            }
            if (!(worst <= 1e-5 * @max(peak, 1e-30))) std.debug.print("[inkling2 proof] visit {d}, {d} accepted: conv rollback off by {e} of {e}\n", .{ v, 1 + acc, worst, peak });
            try testing.expect(worst <= 1e-5 * @max(peak, 1e-30));
        }
    }
}

test "inkling2 proof: the quantized pack is no further from the truth than mlx-lm's GPU run" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tk = try tokens(a);
    defer tk.deinit();
    var p = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fixtureOf("tiny_q"), .{ .gpu = true });
    defer p.close();
    var cache = try test_cache.TestCache.init(a, p.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const n: u32 = @intCast(tk.value.prompt.len);
    const pre = try step(a, &p, &cache, tk.value.prompt, 0, .all_rows, .prefill);
    defer a.free(pre);
    const dec = try a.alloc(f32, tk.value.decode.len * VOCAB);
    defer a.free(dec);
    for (tk.value.decode, 0..) |t, i| {
        const row = try step(a, &p, &cache, &.{t}, n + @as(u32, @intCast(i)), .last_row, .decode);
        defer a.free(row);
        @memcpy(dec[i * VOCAB ..][0..VOCAB], row);
    }
    try expectNoWorse("prefill", pre, refRows("tiny_q/gpu_prefill"), refRows("tiny_q/truth_prefill"));
    try expectNoWorse("decode", dec, refRows("tiny_q/gpu_decode"), refRows("tiny_q/truth_decode"));
    try expectRows(pre, refRows("tiny_q/gpu_prefill"));
    try expectRows(dec, refRows("tiny_q/gpu_decode"));
}

test "inkling2 proof: an image row is refused by name" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var p = try open(arena.allocator());
    defer p.close();
    var cache = try test_cache.TestCache.init(a, p.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const media = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(media);
    var h = [_]abi.Handle{cache.handle()};
    const toks = [_]u32{ 3, 4 };
    var out: abi.ForwardOut = .{};
    const v = p.table.forward(p.model, &.{ .mode = .prefill, .rows = 2, .tokens = .of(&toks), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops, .media = .{ .ctx = media.ctx } }, &out);
    try testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
    try testing.expect(std.mem.indexOf(u8, v.why.slice(), "images") != null);
    try testing.expect(out.logits.ctx == null);
}
