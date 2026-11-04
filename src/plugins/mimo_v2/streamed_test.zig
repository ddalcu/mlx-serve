//! The resident MiMo with its routed experts read from a streamed bank (`experts.bin`, the MXFP4 bank format of
//! MiMo's stream packs) through the MoE block's `Routed` hook, against the same experts in memory: the served logits
//! and the MoE block alone, bit for bit, with every expert resident and with a few rows a layer (evictions every
//! call). Owns tests only.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const moe_fp4 = @import("../../sdk/blocks/moe/moe_fp4.zig");
const streamed = @import("../../sdk/weights/streamed.zig");
const mimo = @import("plugin.zig");
const trunk = @import("proof_test.zig");
const stream = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

const testing = std.testing;
const FIRST_MOE: u32 = 1;
const MOE_LAYERS: u32 = 3;

/// Every MoE layer's experts as MXFP4 records in the bank's layout: random E2M1 words, E8M0 scales of 2^-4 and 2^-3.
const Bank = struct {
    geo: Geometry,
    bytes: []u8,

    const Geometry = if (build_options.mlx_stream) stream.bank_source.Geometry else void;

    fn init(a: std.mem.Allocator, seed: u64, n_experts: u32) !Bank {
        const geo = try Geometry.init(trunk.H, trunk.MOE_INTER, n_experts, FIRST_MOE, MOE_LAYERS);
        const bytes = try a.alloc(u8, geo.total_bytes);
        var rnd = std.Random.DefaultPrng.init(seed);
        for (0..geo.num_moe_layers) |l| for (0..geo.num_experts) |e| for (geo.segments) |seg| {
            const at = bytes[geo.recordOffset(@intCast(FIRST_MOE + l), @intCast(e)) + seg.offset ..][0..seg.length];
            for (at) |*b| b.* = if (seg.dtype == .U32) rnd.random().int(u8) else @as(u8, 123) + rnd.random().int(u1);
        };
        return .{ .geo = geo, .bytes = bytes };
    }

    /// Model layer `layer`'s `mlp.switch_mlp.{gate,up,down}_proj` as stacked MXFP4 banks.
    fn put(b: *const Bank, p: *fixture.Pack, layer: u32) !void {
        const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
        for (projs, 0..) |proj, pi| for ([_][]const u8{ "weight", "scales" }, 0..) |part, ci| {
            const seg = b.geo.segments[pi * 2 + ci];
            const data = try testing.allocator.alloc(u8, b.geo.num_experts * seg.length);
            defer testing.allocator.free(data);
            for (0..b.geo.num_experts) |e| @memcpy(data[e * seg.length ..][0..seg.length], b.bytes[b.geo.recordOffset(layer, @intCast(e)) + seg.offset ..][0..seg.length]);
            const shape = [_]c_int{ @intCast(b.geo.num_experts), @intCast(seg.shape[0]), @intCast(seg.shape[1]) };
            const dt: mlx.mlx_dtype = if (seg.dtype == .U32) .uint32 else .uint8;
            const arr = mlx.mlx_array_new_data(data.ptr, &shape, 3, dt);
            defer _ = mlx.mlx_array_free(arr);
            var buf: [96]u8 = undefined;
            try p.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ layer, proj, part }), arr);
        };
    }
};

/// The proof pack with `n` experts a layer, its banks MXFP4 from `bank`.
fn writePack(p: *fixture.Pack, bank: *const Bank, n: c_int) !void {
    try trunk.writePack(p, false);
    var buf: [96]u8 = undefined;
    for (FIRST_MOE..FIRST_MOE + MOE_LAYERS) |i| {
        try p.dense(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.{s}", .{ i, if (i == 2) "gate.proj.weight" else "gate.weight" }), &.{ n, trunk.H }, 0.1, 0);
        if (i != 3) try p.denseAs(try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{i}), &.{n}, 0.3, 0, .float32);
        try bank.put(p, @intCast(i));
    }
}

fn configJson(a: std.mem.Allocator, n: u32) ![]const u8 {
    const base = trunk.configJson(false);
    const sized = try std.mem.replaceOwned(u8, a, base, "\"n_routed_experts\":8,", try std.fmt.allocPrint(a, "\"n_routed_experts\":{d},", .{n}));
    return std.mem.replaceOwned(u8, a, sized, "\"eos_token_id\":[2,151645]}", "\"eos_token_id\":[2,151645],\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}}");
}

/// Every MoE layer's hook set to `r` (null: the in-memory banks), each decode call predicting the next layer's reads
/// with its router as the plugin installs it.
fn hook(m: *mimo.Loaded, r: ?streamed.Routed) void {
    for (m.layers, 0..) |*L, i| switch (L.mlp) {
        .second => |*w| {
            w.streamed = if (r) |x| .{ .routed = x, .layer = @intCast(i) } else null;
            if (r != null and i + 1 < m.layers.len and m.layers[i + 1].mlp == .second) {
                const nx = &m.layers[i + 1].mlp.second;
                if (nx.router.s.ctx == null) w.streamed.?.next = .{ .router = nx.router.w, .bias = nx.route.sigmoid_bias.expert_bias };
            }
        },
        .first => {},
    };
}

/// A prompt past the window (the wide MoE), decodes (the fp4 decode kernels), a two-row verify (the same kernels at
/// two rows) and an eight-row verify (the sorted gathers); logits host f32, a streamed source settled after each eval.
const seq = [_]struct { mode: abi.Mode, ids: []const u32 }{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200, 9, 33, 77, 140, 2, 18 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 5, 61 } },
    .{ .mode = .verify, .ids = &.{ 6, 62, 201, 5, 12, 91, 14, 8 } },
    .{ .mode = .decode, .ids = &.{64} },
};

fn logits(b: *both.Both, a: std.mem.Allocator, r: ?streamed.Routed) ![]u32 {
    var side: trunk.Side = undefined;
    try side.init();
    defer side.deinit();
    var out: std.ArrayList(u32) = .empty;
    for (seq) |st| {
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        const handles = [_]abi.Handle{side.slot.handle()};
        const got = try trunk.tableLogits(&g, b.xfm.plugin.?.model, .{ .mode = st.mode, .rows = @intCast(st.ids.len), .tokens = .of(st.ids), .positions = .of(&.{@intCast(side.off)}), .caches = .of(&handles) });
        const f = try g.astype(got, .float32);
        try mlx.check(mlx.mlx_array_eval(f));
        if (r) |x| try x.settle();
        const n = mlx.mlx_array_size(f);
        try out.appendSlice(a, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..n])));
        side.off += st.ids.len;
    }
    return out.items;
}

/// Layer `layer`'s MoE block alone on `rows` random rows in `dt`, bits of the f32 output.
fn blockOut(b: *both.Both, a: std.mem.Allocator, w: *const moe.Moe.Weights, rows: c_int, dt: mlx.mlx_dtype, seed: u64, r: ?streamed.Routed) ![]u32 {
    var g = ops.Graph.init(testing.allocator, b.xfm.s);
    defer g.deinit();
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try p.denseAs("x", &.{ 1, rows, trunk.H }, 1.0, 0, dt);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_map_string_to_array_get(&x, p.map, "x"));
    const y = try g.astype(try moe.mlp(w, &g, x), .float32);
    try mlx.check(mlx.mlx_array_eval(y));
    if (r) |s| try s.settle();
    const n = mlx.mlx_array_size(y);
    return a.dupe(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(y) orelse return error.NoData)[0..n])));
}

fn parity(n_experts: u32) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const bank = try Bank.init(a, 0x5713, n_experts);
    var p = fixture.Pack.init(0x5714);
    defer p.deinit();
    try writePack(&p, &bank, @intCast(n_experts));
    var b = try both.Both.openAs(&mimo.table, &p, try configJson(a, n_experts), false, .none);
    defer b.close();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stream.bank_source.writeBank(a, io, tmp.dir, bank.geo, bank.bytes, "synthetic/mimo-tiny", "0000000000000000000000000000000000000000");
    var buf: [512]u8 = undefined;
    const bank_dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const m: *mimo.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    const moe_w = &m.layers[FIRST_MOE].mlp.second;

    // The in-memory banks: the served steps, then the block alone at every arm's widths in bf16 and f32.
    const widths = [_]c_int{ 1, 2, 8, 20, 128 };
    const native = try logits(&b, a, null);
    // 8200 x 2 pairs is past one call's cap (`max_call_ids`): the call splits into even pieces of the same arm.
    const wide_rows = 8200;
    const native_wide = try blockOut(&b, a, moe_w, wide_rows, .float32, 0x77, null);
    var native_block: [2][widths.len][]u32 = undefined;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |dt, di| for (widths, 0..) |rows, wi| {
        native_block[di][wi] = try blockOut(&b, a, moe_w, rows, dt, 0x10 + wi, null);
    };

    const few: [MOE_LAYERS]u32 = @splat(3);
    const most: [MOE_LAYERS]u32 = @splat(@max(3, n_experts * 2 / 3));
    for ([_]?[]const u32{ null, &few, &most }) |rows| {
        var sw = try streamed.Streamed.open(testing.allocator, io, bank_dir, b.xfm.s, .{ .rows = rows });
        defer sw.close();
        const r = sw.routedSource();
        hook(m, r);
        defer hook(m, null);
        const banked = moe_fp4.banked_dispatches;
        try testing.expectEqualSlices(u32, native, try logits(&b, a, r));
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |dt, di| for (widths, 0..) |w, wi| {
            try testing.expectEqualSlices(u32, native_block[di][wi], try blockOut(&b, a, moe_w, w, dt, 0x10 + wi, r));
        };
        try testing.expectEqualSlices(u32, native_wide, try blockOut(&b, a, moe_w, wide_rows, .float32, 0x77, r));
        const st = sw.src.mxfp4.stats();
        std.debug.print("mimo_v2 streamed ({d} experts, {s}): {d} routes, {d} hits, {d} misses, {d} evictions\n", .{ n_experts, if (rows == null) "every expert resident" else if (rows.?[0] == 3) "3 rows a layer" else "2/3 of the experts a layer", st.route_calls, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions });
        // With three rows a layer, decode reads pairs from more than one slot bank.
        if (rows != null and rows.?[0] == 3) try testing.expect(st.expert_cache_evictions > 0 and moe_fp4.banked_dispatches != banked);
        // Short of every expert, decode calls read the next layer's predicted experts ahead.
        if (rows != null) try testing.expect(st.spec_issued > 0);
    }
}

// ── tests ──

test "mimo_v2 streamed: the resident's experts through the Routed hook from an experts.bin are its in-memory banks' logits bit for bit" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    // 8 experts: the prompt's gathers take the row-block class (40 pairs, 5 a row); 32: the per-pair one; 96: a call
    // whose experts outnumber one fetch (`sortedGroups`).
    try parity(8);
    try parity(32);
    try parity(96);
}

/// Every layer's q/k/v and layer 0's dense MLP as fp8res packs them: e4m3 codes (no NaN code) beside an f32 scale per
/// 64 x 128 tile (`fp8_block128`).
fn fp8Residents(p: *fixture.Pack, seed: u64) !void {
    var rnd = std.Random.DefaultPrng.init(seed);
    var buf: [96]u8 = undefined;
    for (0..4) |i| for ([_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |leaf| {
        const base = try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, leaf });
        var old = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(old);
        const z = try std.fmt.allocPrintSentinel(testing.allocator, "{s}.weight", .{base}, 0);
        defer testing.allocator.free(z);
        if (mlx.mlx_map_string_to_array_get(&old, p.map, z.ptr) != 0 or old.ctx == null) continue;
        const sh = mlx.getShape(old);
        const n: usize = @intCast(sh[0]);
        const k: usize = @intCast(sh[1]);
        const codes = try testing.allocator.alloc(u8, n * k);
        defer testing.allocator.free(codes);
        for (codes) |*c| c.* = rnd.random().int(u8) & 0xf7;
        const sc = try testing.allocator.alloc(f32, (n / 64) * (k / 128));
        defer testing.allocator.free(sc);
        for (sc) |*v| v.* = std.math.ldexp(1.0 + rnd.random().float(f32), -12);
        const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ sh[0], sh[1] }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        const g = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ @intCast(n / 64), @intCast(k / 128) }, 2, .float32);
        defer _ = mlx.mlx_array_free(g);
        try p.put(z, w);
        try p.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight_scale_inv", .{ i, leaf }), g);
    };
}

/// `p`'s tensors but the routed experts (`switch_mlp`), as `dir/model.safetensors`.
fn saveResidents(p: *const fixture.Pack, dir: []const u8) !void {
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const it = mlx.mlx_map_string_to_array_iterator_new(p.map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) != 0 or key == null) break;
        if (std.mem.indexOf(u8, std.mem.span(key.?), "switch_mlp") == null) try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.?, v));
    }
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
}

test "mimo_v2 streaming pack: fp8 residents and experts read from its experts.bin serve the in-memory pack's logits bit for bit" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const bank = try Bank.init(a, 0x5715, 8);
    var p = fixture.Pack.init(0x5716);
    defer p.deinit();
    try writePack(&p, &bank, 8);
    try fp8Residents(&p, 0x5717);
    const mem_json = try configJson(a, 8);
    var mem = try both.Both.openAs(&mimo.table, &p, mem_json, false, .none);
    defer mem.close();
    const want = try logits(&mem, a, null);

    // The streaming pack: the residents, the bank and its manifest, and the config's expert_streaming block.
    var tmp = testing.tmpDir(.{});
    var buf: [512]u8 = undefined;
    const dir = try testing.allocator.dupe(u8, buf[0..try tmp.dir.realPath(io, &buf)]);
    try saveResidents(&p, dir);
    try stream.bank_source.writeBank(a, io, tmp.dir, bank.geo, bank.bytes, "synthetic/mimo-tiny", "0000000000000000000000000000000000000000");
    const js = try std.mem.replaceOwned(u8, a, mem_json, "\"quantization\":{", "\"residents_format\":\"fp8_block128\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-mxfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},\"quantization\":{");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = js });
    var cfg = try both.parse(&mimo.table, dir, js);
    const xfm = try driver.load(std.Io.Threaded.global_single_threaded.io(), testing.allocator, &cfg, dir, fixture.stream());
    var b: both.Both = .{ .tmp = tmp, .dir = dir, .cfg = cfg, .xfm = xfm };
    defer b.close();
    const m: *mimo.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expect(m.experts != null and m.layers[FIRST_MOE].mlp.second.streamed != null);
    try testing.expectEqualSlices(u32, want, try logits(&b, a, m.experts.?.routedSource()));
}

/// One expert projection of `bank` (model layer 0) dequantized to f32 `[out, in]`, owned by `g`.
fn dequant(g: *ops.Graph, bank: *const Bank, e: u32, proj: usize) !mlx.mlx_array {
    var parts: [2]mlx.mlx_array = undefined;
    for (&parts, 0..) |*o, ci| {
        const seg = bank.geo.segments[proj * 2 + ci];
        const shape = [_]c_int{ @intCast(seg.shape[0]), @intCast(seg.shape[1]) };
        o.* = try g.adopt(mlx.mlx_array_new_data(bank.bytes[bank.geo.recordOffset(0, e) + seg.offset ..].ptr, &shape, 2, if (seg.dtype == .U32) .uint32 else .uint8));
    }
    return g.call(mlx.mlx_dequantize, .{ parts[0], parts[1], mlx.mlx_array{}, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "mxfp4", mlx.mlx_array{}, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true } });
}

/// Rel RMS error of `got` against `truth`.
fn relRms(got: []const f32, truth: []const f64) f64 {
    var se: f64 = 0;
    var st: f64 = 0;
    for (got, truth) |v, t| {
        se += (v - t) * (v - t);
        st += t * t;
    }
    return @sqrt(se / st);
}

test "mimo_v2 streamed experts at the release's 4096 / 2048: the hook's decode MoE against fp32 truth" {
    if (!build_options.mlx_stream or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const H = 4096;
    const I = 2048;
    const E = 8;
    const K = 8;
    const geo = try Bank.Geometry.init(H, I, E, 0, 1);
    const bank: Bank = .{ .geo = geo, .bytes = try a.alloc(u8, geo.total_bytes) };
    var rnd = std.Random.DefaultPrng.init(0x4096);
    for (0..E) |e| for (geo.segments) |seg| {
        const at = bank.bytes[geo.recordOffset(0, @intCast(e)) + seg.offset ..][0..seg.length];
        for (at) |*b| b.* = if (seg.dtype == .U32) rnd.random().int(u8) else @as(u8, 119) + rnd.random().int(u1);
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stream.bank_source.writeBank(a, io, tmp.dir, geo, bank.bytes, "synthetic/mimo-release", "0000000000000000000000000000000000000000");
    var buf: [512]u8 = undefined;
    const s = fixture.stream();
    var sw = try streamed.Streamed.open(testing.allocator, io, buf[0..try tmp.dir.realPath(io, &buf)], s, .{});
    defer sw.close();
    const src = sw.src.mxfp4;
    const qp: @import("../../sdk/quant/params.zig").QuantParams = .{ .bits = 4, .group_size = 32, .mode = .mxfp4 };
    const w: moe.Moe.Weights = .{
        .router = .{},
        .route = .softmax,
        .gate = .{ .w = .{}, .qp = qp },
        .up = .{ .w = .{}, .qp = qp },
        .down = .{ .w = .{}, .qp = qp },
        .top_k = K,
        .experts = E,
        .streamed = .{ .routed = sw.routedSource(), .layer = 0 },
    };

    // One token and two (the fp4 decode kernels), four (the hook's sorted gathers; the stream's union texts).
    for ([_]usize{ 1, 2, 4 }) |n| {
        var g = ops.Graph.init(testing.allocator, s);
        defer g.deinit();
        const xs = try a.alloc(f32, n * H);
        for (xs) |*v| v.* = 0.05 * rnd.random().floatNorm(f32);
        const ids = try a.alloc(u32, n * K);
        const wts = try a.alloc(f32, n * K);
        for (0..n) |t| {
            var perm: [E]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
            rnd.random().shuffle(u32, &perm);
            var sum: f32 = 0;
            for (0..K) |k| {
                ids[t * K + k] = perm[k];
                wts[t * K + k] = 0.1 + rnd.random().float(f32);
                sum += wts[t * K + k];
            }
            for (wts[t * K ..][0..K]) |*v| v.* /= sum;
        }
        const nc: c_int = @intCast(n);
        const x = try g.astype(try g.fromSlice(f32, xs, &.{ nc, H }), .bfloat16);
        const wf = try g.fromSlice(f32, wts, &.{ nc, K });

        // The hook: the resident's arm on the slot banks (its routing weights in the activation dtype).
        const hook_y = try g.astype(try moe.routedSum(&w, &g, try g.reshape(x, &.{ 1, nc, H }), try g.fromSlice(u32, ids, &.{ 1, nc, K }), try g.astype(try g.reshape(wf, &.{ 1, nc, K }), .bfloat16)), .float32);
        try mlx.check(mlx.mlx_array_eval(hook_y));
        try src.settle();
        const hooked = try a.dupe(f32, (mlx.mlx_array_data_float32(hook_y) orelse return error.NoData)[0 .. n * H]);

        // fp32 truth: each expert dequantized, silu(gate x) * up x, down, weighted in f64 on the host.
        const truth = try a.alloc(f64, n * H);
        @memset(truth, 0);
        const xf = try g.astype(x, .float32);
        for (0..E) |e| {
            var eg = ops.Graph.init(testing.allocator, s);
            defer eg.deinit();
            const tr = struct {
                fn f(gg: *ops.Graph, m: mlx.mlx_array) !mlx.mlx_array {
                    return gg.call(mlx.mlx_transpose, .{m});
                }
            }.f;
            const gt = try eg.matmul(xf, try tr(&eg, try dequant(&eg, &bank, @intCast(e), 0)));
            const up = try eg.matmul(xf, try tr(&eg, try dequant(&eg, &bank, @intCast(e), 1)));
            const act = try eg.mul(try eg.mul(gt, try eg.sigmoid(gt)), up);
            const dn = try eg.matmul(act, try tr(&eg, try dequant(&eg, &bank, @intCast(e), 2)));
            try mlx.check(mlx.mlx_array_eval(dn));
            const d = (mlx.mlx_array_data_float32(dn) orelse return error.NoData)[0 .. n * H];
            for (0..n) |t| for (0..K) |k| if (ids[t * K + k] == e) {
                for (0..H) |j| truth[t * H + j] += @as(f64, wts[t * K + k]) * d[t * H + j];
            };
        }
        // A bf16-output path of the function: within bf16's class of the truth.
        try testing.expect(relRms(hooked, truth) < 1e-2);
    }
}
