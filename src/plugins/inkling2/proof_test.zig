//! The inkling2 table against itself on its tiny looped pack: a prompt forwarded whole, in chunks and token by token
//! serves the same rows, past the sliding window, the relative bias's extent and the log-scaling floor, so every
//! visit's KV and conv windows carry, the one-row sink path agrees with the wide one and the loop's passes keep
//! their own state. On the GPU stream the one-row step runs its kernels (`kernels.zig`), so the op chain of the wide
//! forward is their reference. Not a numeric proof against mlx-lm: that oracle (`models/inkling2`, an open PR) is
//! still owed.

const std = @import("std");
const plugin = @import("plugin.zig");
const draft = @import("draft.zig");
const dflash2 = @import("dflash2.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const Tensors = @import("../../sdk/testing/tensors.zig").Tensors;
const parity = @import("../../sdk/testing/parity.zig");
const run = @import("../../sdk/bases/run.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");

const testing = std.testing;
const VOCAB = 60;
const prompt = [_]u32{ 1, 17, 42, 55, 3, 9, 7, 30, 44, 5, 22, 31, 58, 2, 13, 49, 27, 8, 36, 19, 51, 4, 40, 11 };

/// One forward of `tokens` from `pos` through the slot's cache; its logits rows as f32.
fn step(a: std.mem.Allocator, pack: *conformance.Loaded, cache: *test_cache.TestCache, tokens: []const u32, pos: usize, logits: abi.Logits) ![]f32 {
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const positions = [_]u32{@intCast(pos)};
    var out: abi.ForwardOut = .{};
    const mode: abi.Mode = if (tokens.len == 1) .decode else .prefill;
    const v = pack.table.forward(pack.model, &.{ .mode = mode, .logits = logits, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&slot), .positions = .of(&positions), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

/// Bar: the stream is bf16 between visits, so two paths whose f32 sums differ in their last bits (a kernel against its
/// op chain) round some elements apart: cosine above 0.999 and no row whose margin decides it picks another token.
fn expectRows(ours: []const f32, ref: []const f32) !void {
    try testing.expectEqual(ref.len, ours.len);
    const rows = ref.len / VOCAB;
    const cmp = parity.qwen4CompareRows(ours, ref, rows, VOCAB);
    const misses = parity.decidedMisses(ours, ref, rows, VOCAB);
    if (cmp.min_cos <= 0.999 or misses > 0) std.debug.print("[inkling2 proof] {d} rows: min cos {d:.6}, decided misses {d}\n", .{ rows, cmp.min_cos, misses });
    try testing.expect(cmp.min_cos > 0.999 and misses == 0);
}

const Setup = struct {
    arena: std.heap.ArenaAllocator,
    pack: conformance.Loaded,
    whole: []f32,

    /// The pack, on the CPU's stream or the GPU's, and the whole prompt's rows from one forward.
    fn init(a: std.mem.Allocator, gpu: bool) !Setup {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .gpu = gpu });
        errdefer pack.close();
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        return .{ .arena = arena, .pack = pack, .whole = try step(a, &pack, &cache, &prompt, 0, .all_rows) };
    }

    fn deinit(s: *Setup, a: std.mem.Allocator) void {
        a.free(s.whole);
        s.pack.close();
        s.arena.deinit();
    }
};

/// The streams a proof runs on: the CPU's, and the GPU's where there is a real one.
fn streams() []const bool {
    return if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) &.{false} else &.{ false, true };
}

// Bar: a last-row prompt forward (`Caps.prompt_last_row`, its head over the last rows) serves the bytes of that row of
// the all-rows forward, under the head's tail rows and past them, on the GPU's stream, where the model is served.
test "inkling2 proof: a last-row prompt serves the last row of the all-rows forward, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var tokens: [300]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast((i * 37 + 11) % VOCAB);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .gpu = true });
    defer pack.close();
    for ([_]usize{ 1, 31, 33, 300 }) |n| {
        var c_all = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer c_all.deinit();
        var c_last = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer c_last.deinit();
        const all = try step(a, &pack, &c_all, tokens[0..n], 0, .all_rows);
        defer a.free(all);
        const last = try step(a, &pack, &c_last, tokens[0..n], 0, .last_row);
        defer a.free(last);
        const want = all[(n - 1) * VOCAB .. n * VOCAB];
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(last))) std.debug.print("[inkling2 proof] last row differs at {d} rows\n", .{n});
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(last));
    }
}

// Bar: on the GPU, the pack's own lm_head over a prompt's last `head_tail_rows` rows gives that row the all-rows bits,
// through the head's own matmul and this build's kernels.
test "inkling2 pack: the lm_head over the last head_tail_rows rows serves the last row of all of them, bit for bit (INKLING2_PACK)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const dir = std.mem.span(@import("../../env.zig").named("INKLING2_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var pack_dir = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer pack_dir.close(io);
    const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
    const index = try std.json.parseFromSliceLeaky(Index, ar, try pack_dir.readFileAlloc(io, "model.safetensors.index.json", ar, .limited(1 << 24)), .{ .ignore_unknown_fields = true });
    const Quant = struct { group_size: u32, bits: u32 };
    const Config = struct { quantization: Quant };
    const q = (try std.json.parseFromSliceLeaky(Config, ar, try pack_dir.readFileAlloc(io, "config.json", ar, .limited(1 << 20)), .{ .ignore_unknown_fields = true })).quantization;
    const shard = index.weight_map.map.get("lm_head.weight") orelse return error.SkipZigTest;

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&map, &meta, try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, shard }, 0), mlx.weightLoadStream()));
    var lin: layers.Linear = .{ .bits = q.bits, .group_size = q.group_size };
    for ([_]*ops.Array{ &lin.w, &lin.s, &lin.b }, [_][:0]const u8{ "lm_head.weight", "lm_head.scales", "lm_head.biases" }) |p, name| {
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_map_string_to_array_get(&x, map, name));
        p.* = try g.adopt(x);
    }
    const hidden: c_int = ops.shape(lin.s)[1] * @as(c_int, @intCast(q.group_size));

    const most = 4100;
    var prng = std.Random.DefaultPrng.init(7);
    const buf = try a.alloc(f32, @intCast(most * hidden));
    defer a.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const rows = try g.astype(try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, most, hidden }, 3, .float32)), .bfloat16);
    for ([_]c_int{ 33, 300, 1536, most }) |n| {
        const x = try g.sliceAxis(rows, 1, 0, n);
        const all = try g.contiguous(try g.astype(try g.sliceAxis(try head_block.project(&g, &lin, x), 1, n - 1, n), .float32));
        const tail = try g.contiguous(try g.astype(try g.sliceAxis(try head_block.project(&g, &lin, try g.sliceAxis(x, 1, n - run.head_tail_rows, n)), 1, run.head_tail_rows - 1, run.head_tail_rows), .float32));
        try ops.evalMany(&.{ all, tail });
        const vocab: usize = @intCast(ops.shape(all)[2]);
        const want = mlx.mlx_array_data_float32(all).?[0..vocab];
        const got = mlx.mlx_array_data_float32(tail).?[0..vocab];
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) std.debug.print("[inkling2 pack] last row differs at {d} rows\n", .{n});
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    }
}

// On the CPU's stream, where the step's chain and the wide forward's are the same sums: the GPU's f32 matmul is good
// to about 1e-3, and on this random pack one such bit flips an expert pick somewhere along twelve steps.
test "inkling2 proof: a prompt in chunks and then token by token serves the rows of one forward" {
    const a = testing.allocator;
    var s = try Setup.init(a, false);
    defer s.deinit(a);
    var cache = try test_cache.TestCache.init(a, s.pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    var at: usize = 0;
    for ([_]usize{ 5, 7 }) |n| {
        const rows = try step(a, &s.pack, &cache, prompt[at .. at + n], at, .all_rows);
        defer a.free(rows);
        try expectRows(rows, s.whole[at * VOCAB .. (at + n) * VOCAB]);
        at += n;
    }
    while (at < prompt.len) : (at += 1) {
        const row = try step(a, &s.pack, &cache, prompt[at .. at + 1], at, .last_row);
        defer a.free(row);
        try expectRows(row, s.whole[at * VOCAB .. (at + 1) * VOCAB]);
    }
}

// Bar: a quantized pack's joined projections are whole 64-row tiles (the tensor unit's quantized kernels take no other
// width), and a part read through the padded join is the part's own projection.
test "inkling2 pack: a quantized joined projection is whole tiles and serves each part (INKLING2_PACK)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const dir = std.mem.span(@import("../../env.zig").named("INKLING2_PACK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rel = try Release.open(a, dir);
    defer rel.close();
    fx_stream = rel.m.base.s;
    var g = ops.Graph.init(a, rel.m.base.s);
    defer g.deinit();
    var joins: usize = 0;
    for (rel.m.layers) |*L| for ([_]*const @TypeOf(L.join_q){ &L.join_q, &L.join_kv }) |j| {
        if (!j.built() or j.lin.s.ctx == null) continue;
        joins += 1;
        try testing.expectEqual(@as(c_int, 0), @rem(ops.shape(j.lin.w)[0], 64));
    };
    try testing.expect(joins > 0);
    const L = &rel.m.layers[0];
    for ([_]c_int{ 1, 12 }) |rows| {
        const x = try randomRow(&g, a, 500 + @as(u64, @intCast(rows)), &.{ 1, rows, 2304 }, 0.5, .bfloat16);
        const y = try layers.projectRows(&g, &L.join_q.lin, x);
        for ([_]*const layers.Linear{ &L.q, &L.gate, &L.r, &L.sink }, 0..) |part, i| {
            try expectClose(try L.join_q.part(&g, y, i), try layers.projectRows(&g, part, x), 2e-2, "joined part");
        }
    }
}

/// The fixture's tensors (mlx-lm's layout) as oMLX's oQ layout stores them: renamed, the projections off one input
/// joined along their rows, a looped layer's gates and message queries stacked under the layer.
fn oqLayout(a: std.mem.Allocator, tensors: []const fixture.Tensor) ![]fixture.Tensor {
    const H = 64;
    const RANK = 8;
    const lm = "model.language_model.";
    const Pack = struct {
        a: std.mem.Allocator,
        by: std.StringHashMap(fixture.Tensor),
        out: std.ArrayList(fixture.Tensor) = .empty,

        fn get(p: *@This(), layer: ?usize, leaf: []const u8) !?fixture.Tensor {
            const name = if (layer) |i| try std.fmt.allocPrint(p.a, lm ++ "layers.{d}.{s}", .{ i, leaf }) else try std.fmt.allocPrint(p.a, lm ++ "{s}", .{leaf});
            return p.by.get(name);
        }

        fn put(p: *@This(), layer: ?usize, leaf: []const u8, shape: []const u64, data: []const f32) !void {
            const name = if (layer) |i| try std.fmt.allocPrint(p.a, "model.layers.{d}.{s}", .{ i, leaf }) else try p.a.dupe(u8, leaf);
            try p.out.append(p.a, .{ .name = name, .shape = try p.a.dupe(u64, shape), .data = data });
        }

        /// `leaves` joined along their rows as `dst`.
        fn join(p: *@This(), layer: usize, dst: []const u8, leaves: []const []const u8) !void {
            var data: std.ArrayList(f32) = .empty;
            for (leaves) |leaf| try data.appendSlice(p.a, (try p.get(layer, leaf)).?.data);
            try p.put(layer, dst, &.{ data.items.len / H, H }, data.items);
        }
    };
    var p: Pack = .{ .a = a, .by = .init(a) };
    for (tensors) |t| try p.by.put(t.name, t);
    try p.put(null, "lm_head.weight", &.{ 64, H }, p.by.get("lm_head.weight").?.data);
    inline for (.{ .{ "embed_tokens.weight", "model.embed.weight" }, .{ "embed_norm.weight", "model.embed_norm.weight" }, .{ "norm.weight", "model.norm.weight" } }) |r| {
        const t = (try p.get(null, r[0])).?;
        try p.put(null, r[1], t.shape, t.data);
    }
    try p.put(null, "model.final_input.query", &.{ 1, H }, (try p.get(null, "final_input.query")).?.data);
    const message = (try p.get(null, "message_input.query")).?.data;
    const renames = [_][2][]const u8{
        .{ "attention_input.query", "attn_input.query" },             .{ "mlp_input.query", "mlp_input.query" },
        .{ "input_layernorm.weight", "attn_norm.weight" },            .{ "q_input_layernorm.weight", "q_input_norm.weight" },
        .{ "k_input_layernorm.weight", "k_input_norm.weight" },       .{ "v_input_layernorm.weight", "v_input_norm.weight" },
        .{ "post_attention_layernorm.weight", "mlp_norm.weight" },    .{ "self_attn.o_proj.weight", "attn.o_proj.weight" },
        .{ "self_attn.q_l2_clip.scale", "attn.q_scale" },             .{ "self_attn.rel_l2_clip.scale", "attn.r_scale" },
        .{ "self_attn.rel_logits_proj.proj", "attn.rel_proj" },       .{ "self_attn.k_sconv.static_weight", "attn.k_static" },
        .{ "self_attn.v_sconv.static_weight", "attn.v_static" },      .{ "self_attn.k_sconv.conv1d.weight", "attn.k_conv" },
        .{ "self_attn.v_sconv.conv1d.weight", "attn.v_conv" },        .{ "post_attention_convolution.conv1d.weight", "attn_conv" },
        .{ "post_mlp_convolution.conv1d.weight", "mlp_conv" },        .{ "mlp.gate.weight", "mlp.router_weight" },
        .{ "mlp.gate.e_score_correction_bias", "mlp.router_bias" },   .{ "mlp.experts.gate_up_proj.weight", "mlp.experts.w13.weight" },
        .{ "mlp.experts.down_proj.weight", "mlp.experts.w2.weight" },
    };
    for (0..8) |i| {
        for (renames) |r| if (try p.get(i, r[0])) |t| {
            // oQ drops a kernel's trailing axis and keeps a lone query as one row.
            const shape: []const u64 = if (std.mem.endsWith(u8, r[0], "conv1d.weight")) t.shape[0..2] else if (t.shape.len == 1 and std.mem.endsWith(u8, r[0], "query")) &.{ 1, H } else t.shape;
            try p.put(i, r[1], shape, t.data);
        };
        const A = "self_attn.";
        if (try p.get(i, "input_layernorm.weight") != null) {
            try p.join(i, "attn.in_proj.weight", &.{ A ++ "q_proj.weight", A ++ "k_proj.weight", A ++ "v_proj.weight", A ++ "r_proj.weight", A ++ "gate_proj.weight", A ++ "sink_proj.weight" });
        } else {
            try p.join(i, "attn.q_proj.weight", &.{ A ++ "q_proj.weight", A ++ "r_proj.weight", A ++ "gate_proj.weight", A ++ "sink_proj.weight" });
            try p.join(i, "attn.k_proj.weight", &.{ A ++ "k_proj.weight", A ++ "k_sconv.weight_proj.weight" });
            try p.join(i, "attn.v_proj.weight", &.{ A ++ "v_proj.weight", A ++ "v_sconv.weight_proj.weight" });
        }
        if (try p.get(i, "q_message_gate.down_proj.weight") == null) continue;
        const down = try a.alloc(f32, 4 * 2 * H * RANK);
        const both = try a.alloc(f32, 4 * RANK * 2 * H);
        inline for (.{ "q", "k", "v", "mlp" }, 0..) |x, g| {
            const d = (try p.get(i, x ++ "_message_gate.down_proj.weight")).?.data;
            const gain = (try p.get(i, x ++ "_message_gate.message_gate_proj.weight")).?.data;
            const forget = (try p.get(i, x ++ "_message_gate.forget_gate_proj.weight")).?.data;
            for (0..RANK) |r| for (0..2 * H) |c| {
                down[((g * 2 + c / H) * H + c % H) * RANK + r] = d[r * 2 * H + c];
                both[(g * RANK + r) * 2 * H + c] = if (c < H) gain[c * RANK + r] else forget[(c - H) * RANK + r];
            };
        }
        try p.put(i, "loop_down", &.{ 4, 2, H, RANK }, down);
        try p.put(i, "loop_gain_forget", &.{ 4, RANK, 2 * H }, both);
        try p.put(i, "loop_message.query", &.{ 4, H }, message[(i - 2) * 4 * H ..][0 .. 4 * H]);
    }
    return p.out.items;
}

/// `tensors` as a quantized pack stores them: the attention projections, the embedding and the head affine in groups
/// of 32 (4 bits in layers 1 and 4, 8 elsewhere; bf16 scales and biases), the norms and the message gates' down
/// projections bf16 (oQ's `loop_down` holds those bf16 values as f32), the rest f32.
fn quantizedPack(a: std.mem.Allocator, tensors: []const fixture.Tensor) ![]u8 {
    var t = Tensors.init(a, 0);
    for (tensors) |x| {
        const n = x.name;
        const attn_proj = std.mem.indexOf(u8, n, "attn.") != null and std.mem.endsWith(u8, n, "_proj.weight");
        if (attn_proj or std.mem.endsWith(u8, n, "embed.weight") or std.mem.endsWith(u8, n, "embed_tokens.weight") or std.mem.eql(u8, n, "lm_head.weight")) {
            const bits: u32 = if (std.mem.indexOf(u8, n, "layers.1.") != null or std.mem.indexOf(u8, n, "layers.4.") != null) 4 else 8;
            const levels: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
            const codes = try a.alloc(u32, x.data.len * bits / 32);
            @memset(codes, 0);
            const scales = try a.alloc(f32, x.data.len / 32);
            const biases = try a.alloc(f32, x.data.len / 32);
            for (scales, biases, 0..) |*s, *b, g| {
                const v = x.data[g * 32 ..][0..32];
                b.* = std.mem.min(f32, v);
                s.* = (std.mem.max(f32, v) - b.*) / levels;
                for (v, g * 32..) |e, k| codes[k * bits / 32] |= @as(u32, @intFromFloat(@min(levels, @round((e - b.*) / s.*)))) << @intCast(k * bits % 32);
            }
            const base = n[0 .. n.len - ".weight".len];
            try t.put("{s}.weight", .{base}, .U32, &.{ x.shape[0], x.shape[1] * bits / 32 }, std.mem.sliceAsBytes(codes));
            try t.values("{s}.scales", .{base}, &.{ x.shape[0], x.shape[1] / 32 }, .BF16, scales);
            try t.values("{s}.biases", .{base}, &.{ x.shape[0], x.shape[1] / 32 }, .BF16, biases);
        } else if (std.mem.endsWith(u8, n, "norm.weight") or std.mem.endsWith(u8, n, "message_gate.down_proj.weight")) {
            try t.values("{s}", .{n}, x.shape, .BF16, x.data);
        } else if (std.mem.endsWith(u8, n, "loop_down")) {
            // Rounded to bf16 (nearest even), as the bf16 file holds it.
            const wide = try a.alloc(f32, x.data.len);
            for (wide, x.data) |*w, e| {
                const u: u32 = @bitCast(e);
                w.* = @bitCast((u + 0x7fff + ((u >> 16) & 1)) & 0xffff_0000);
            }
            try t.values("{s}", .{n}, x.shape, .F32, wide);
        } else try t.values("{s}", .{n}, x.shape, .F32, x.data);
    }
    return t.bytes();
}

/// The whole prompt's rows from the fixture with `weights` (safetensors bytes) in place of its own.
fn wholeRows(a: std.mem.Allocator, gpu: bool, weights: []const u8) ![]f32 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var fx = try plugin.conformanceFixture(arena.allocator());
    const files = try arena.allocator().dupe(fixture.File, fx.files);
    files[1].bytes = weights;
    fx.files = files;
    var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fx, .{ .gpu = gpu });
    defer pack.close();
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    return step(a, &pack, &cache, &prompt, 0, .all_rows);
}

// Bar: the fixture as oMLX's oQ layout stores it serves the bytes the mlx-lm layout serves for a prompt's rows, dense
// and quantized as packs ship, on the CPU's stream and the GPU's.
test "inkling2 proof: oMLX's oQ layout serves the mlx-lm layout's rows, bit for bit" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const lm = try plugin.fixtureTensors(ar);
    const oq = try oqLayout(ar, lm);
    const packs = [_][2][]const u8{
        .{ try fixture.safetensors(ar, lm), try fixture.safetensors(ar, oq) },
        .{ try quantizedPack(ar, lm), try quantizedPack(ar, oq) },
    };
    for (streams()) |gpu| for (packs) |p| {
        const want = try wholeRows(a, gpu, p[0]);
        defer a.free(want);
        const got = try wholeRows(a, gpu, p[1]);
        defer a.free(got);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    };
}

/// The release parsed and loaded through the plugin table (its defaults), for the tests that run it.
const Release = struct {
    weights: @import("../../sdk/weights/resident.zig").Resident,
    parsed: abi.Parsed = .{},
    m: *plugin.Loaded,
    s: mlx.mlx_stream,

    fn open(a: std.mem.Allocator, pack_dir: []const u8) !Release {
        const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
        const io = std.Io.Threaded.global_single_threaded.io();
        var fs = try FsPackDir.open(io, pack_dir);
        defer fs.close();
        const pd = fs.packDir();
        const cfg_json = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{pack_dir}), a, .limited(1 << 20));
        const Resident = @import("../../sdk/weights/resident.zig").Resident;
        const loadWeightsOpt = @import("../../sdk/weights/load.zig").loadWeightsOpt;
        var r: Release = .{ .weights = Resident.adopt(try loadWeightsOpt(io, std.heap.c_allocator, pack_dir, .{})), .m = undefined, .s = mlx.mlx_default_gpu_stream_new() };
        const t = &plugin.table;
        if (!t.parse(&pd, .of(cfg_json), .{}, null, false, &r.parsed).isOk()) return error.ParseRefused;
        const src = r.weights.source();
        var why: abi.Verdict = .{};
        const mp = t.load(&.{ .model_dir = .of(pack_dir), .dir = &pd, .parsed = &r.parsed, .stream = .{ .ctx = r.s.ctx }, .weights = &src }, &why) orelse return error.LoadRefused;
        r.m = @ptrCast(@alignCast(mp));
        return r;
    }

    fn close(r: *Release) void {
        _ = mlx.mlx_synchronize(r.s);
        plugin.table.unload(r.m);
        plugin.table.free_parsed(&r.parsed);
        r.weights.deinit();
        _ = mlx.mlx_stream_free(r.s);
    }
};

fn argmaxOf(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return @intCast(best);
}

/// The release tokenizer's ids of "def scale_quokka_ledger(entries, factor):\n    total = ...\n\n", of a filler sentence, and of
/// the function's first line again.
const quokka_def = [_]u32{ 1719, 152369, 62, 369, 140073, 62, 46025, 7, 80425, 11, 150935, 8, 25, 198, 149862, 150269, 196977, 150811, 7, 68, 13, 4605, 149887, 149888, 149881, 157358, 8, 198, 149862, 150583, 196989, 68, 13, 4605, 196987, 150935, 196990, 150269, 196986, 220, 15, 13, 80940, 149887, 149888, 149881, 157358, 60, 68014 };
const quokka_filler = [_]u32{ 371, 158897, 155511, 151470, 149876, 151346, 153194, 149880, 149876, 152504, 3661, 149894, 149876, 159167, 13, 220 };
const quokka_again = [_]u32{ 1719, 152369, 62, 369, 140073, 62, 46025, 7, 80425, 11, 150935, 8, 25, 198 };

// Bar: on the release (`INKLING2_RELEASE`), a function defined further back than the drafter's window and then named
// again: the first round's draft is not the one a read of only the newest window - 1 positions gives, and greedy
// rounds (a verify, then its finish) commit the plain greedy decode's tokens up to the first difference, and there a
// near-tie of the plain step's logits (the verify sums in another order than a decode row), within 1% of its top.
test "inkling2 release: the drafter's full layer reads the prompt past its window" {
    const env = @import("../../env.zig");
    const pack_dir = std.mem.span(env.get(.INKLING2_RELEASE) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rel = try Release.open(a, pack_dir);
    defer rel.close();
    // Apple9+ only (`draftsOn`).
    const d = rel.m.drafter orelse return error.SkipZigTest;
    try testing.expect(rel.m.spec.full != null and rel.m.spec.full.? == d);
    const t = &plugin.table;
    const nd = t.spec.?.native_drafter.?;
    const n_layers = rel.parsed.shell.num_hidden_layers;
    // The release tokenizer's ids of "def scale_quokka_ledger(entries, factor):\n    total = ...\n\n", of a filler
    // sentence, and of the function's first line again.
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, &quokka_def);
    while (ids.items.len < quokka_def.len + d.cfg.window) try ids.appendSlice(a, &quokka_filler);
    try ids.appendSlice(a, &quokka_again);
    const n: u32 = @intCast(ids.items.len);
    const prefill = struct {
        fn run(m: *plugin.Loaded, al: std.mem.Allocator, cache: *test_cache.TestCache, tokens: []const u32, slot: u32) !u32 {
            var h = [_]abi.Handle{cache.handle()};
            var out: abi.ForwardOut = .{};
            try testing.expect(plugin.table.forward(m, &.{ .mode = .prefill, .logits = .last_row, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&[_]u32{slot}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
            return argmaxOf(try conformance.consumeF32(al, out.logits));
        }
    };
    // Slot 0 the plain decode, slots 1 and 2 the rounds with and without the full read, each its own drafter context.
    var plain = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
    defer plain.deinit();
    var want: [64]u32 = undefined;
    var step_logits: [want.len][]f32 = undefined;
    want[0] = try prefill.run(rel.m, a, &plain, ids.items, 0);
    for (1..want.len) |i| {
        var h = [_]abi.Handle{plain.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(t.forward(rel.m, &.{ .mode = .decode, .logits = .last_row, .rows = 1, .tokens = .of(want[i - 1 .. i]), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{n + @as(u32, @intCast(i)) - 1}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
        step_logits[i] = try conformance.consumeF32(a, out.logits);
        want[i] = argmaxOf(step_logits[i]);
    }
    var first_draft: [2][]const u32 = undefined;
    for ([_]bool{ true, false }, 0..) |full, arm| {
        const name = if (full) "full read" else "newest window only";
        rel.m.spec.full = if (full) d else null;
        defer rel.m.spec.full = d;
        const slot: u32 = @intCast(1 + arm);
        var cache = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        defer cache.deinit();
        var t1 = try prefill.run(rel.m, a, &cache, ids.items, slot);
        var got: std.ArrayList(u32) = .empty;
        var pos: u32 = n;
        var verified: [abi.max_round_tokens]u32 = @splat(0);
        var kept: [abi.max_round_tokens]u32 = @splat(0);
        for (0..8) |r| {
            var h = [_]abi.Handle{cache.handle()};
            var out: abi.ForwardOut = .{};
            try testing.expect(nd.verify.?(rel.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&[_]u32{t1}), .slots = .of(&[_]u32{slot}), .positions = .of(&[_]u32{pos}), .caches = .of(&h), .cache_ops = &test_cache.ops, .accept_cap = 7 }, &out).isOk());
            const rows: usize = out.round.n;
            const logits = try conformance.consumeF32(a, out.logits);
            const vocab = logits.len / rows;
            var accepted: usize = 0;
            while (accepted + 1 < rows and argmaxOf(logits[accepted * vocab ..][0..vocab]) == out.round.tokens[accepted + 1]) accepted += 1;
            for (1..rows) |k| verified[k] += 1;
            for (1..accepted + 1) |k| kept[k] += 1;
            if (r == 0) first_draft[arm] = try a.dupe(u32, out.round.tokens[1..rows]);
            try got.appendSlice(a, out.round.tokens[0 .. 1 + accepted]);
            t1 = argmaxOf(logits[accepted * vocab ..][0..vocab]);
            var fin: abi.ForwardOut = .{};
            try testing.expect(nd.finish.?(rel.m, @intCast(accepted), &fin).isOk());
            pos = @intCast(fin.round.position);
        }
        std.debug.print("inkling2 release: {s}, a prompt of {d}: first draft {any}; draft positions 1 to {d} verified {any}, accepted {any}\n", .{ name, n, first_draft[arm], d.cfg.block - 1, verified[1..d.cfg.block], kept[1..d.cfg.block] });
        const m = @min(got.items.len, want.len);
        var same: usize = 0;
        while (same < m and want[same] == got.items[same]) same += 1;
        if (same < m) {
            // Only a step past the first: the first token comes from each slot's own prompt forward.
            try testing.expect(same >= 1);
            const lg = step_logits[same];
            const gap = lg[want[same]] - lg[got.items[same]];
            std.debug.print("inkling2 release: {s} rounds and plain decode part at token {d} ({d} vs {d}): the plain step's logit gap {d:.4} of its top {d:.3}\n", .{ name, same, want[same], got.items[same], gap, lg[want[same]] });
            try testing.expect(gap <= 0.01 * @abs(lg[want[same]]));
        }
    }
    try testing.expect(!std.mem.eql(u32, first_draft[0], first_draft[1]));

    // A copy round (`Forward.draft_ids`) verifies the given drafts in place of the drafter's: handed the plain decode's
    // own continuation, it keeps it, up to a near tie of the plain step's logits as above.
    {
        var cache = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        defer cache.deinit();
        const t1 = try prefill.run(rel.m, a, &cache, ids.items, 3);
        try testing.expectEqual(want[0], t1);
        var h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(nd.step(rel.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&[_]u32{t1}), .slots = .of(&[_]u32{3}), .positions = .of(&[_]u32{n}), .caches = .of(&h), .cache_ops = &test_cache.ops, .accept_cap = 15, .draft_ids = .of(want[1..16]) }, &out).isOk());
        try testing.expectEqual(@as(u32, 16), out.round.rows);
        const kept = out.round.accepted;
        std.debug.print("inkling2 release: a copy of the plain continuation keeps {d} of 15\n", .{kept});
        try testing.expectEqualSlices(u32, want[0 .. 1 + kept], out.round.tokens[0 .. 1 + kept]);
        if (kept < 15) {
            const lg = step_logits[kept + 1];
            try testing.expect(lg[want[kept + 1]] - lg[out.round.next_token] <= 0.01 * @abs(lg[want[kept + 1]]));
        }
    }
    // A sampled round's verify draws its drafts from the drafter's law and hands that law back, a row a draft.
    {
        var cache = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        defer cache.deinit();
        const t1 = try prefill.run(rel.m, a, &cache, ids.items, 4);
        var h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(nd.verify.?(rel.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&[_]u32{t1}), .slots = .of(&[_]u32{4}), .positions = .of(&[_]u32{n}), .caches = .of(&h), .cache_ops = &test_cache.ops, .accept_cap = 7, .draft_temperature = 1.0, .draft_seed = 11 }, &out).isOk());
        const drafts = out.round.n - 1;
        a.free(try conformance.consumeF32(a, out.logits));
        if (drafts > 0) {
            const law = try conformance.consumeF32(a, out.draft_probs);
            const vocab = law.len / drafts;
            for (0..drafts) |i| {
                const row = law[i * vocab ..][0..vocab];
                var sum: f32 = 0;
                for (row) |x| sum += x;
                try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-3);
                try testing.expect(row[out.round.tokens[1 + i]] > 0);
            }
        }
        var fin: abi.ForwardOut = .{};
        try testing.expect(nd.finish.?(rel.m, 0, &fin).isOk());
    }
}

// Bar: on the release beside its DFlash2 draft (`INKLING2_DFLASH2_PACK`: the release's files and `drafter/`), greedy
// rounds (a verify, then its finish) commit the plain greedy decode's tokens up to the first difference, and there a
// near-tie of the plain step's logits; and a sampled round hands back a law of the selector's candidates, each row
// summing to one.
test "inkling2 release: a DFlash2 draft's greedy rounds commit the plain decode and a sampled round hands back its law" {
    const env = @import("../../env.zig");
    const pack_dir = std.mem.span(env.get(.INKLING2_DFLASH2_PACK) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rel = try Release.open(a, pack_dir);
    defer rel.close();
    // Apple9+ only (`draftsOn`).
    const d = rel.m.drafter2 orelse return error.SkipZigTest;
    const t = &plugin.table;
    const nd = t.spec.?.native_drafter.?;
    const n_layers = rel.parsed.shell.num_hidden_layers;
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, &quokka_def);
    try ids.appendSlice(a, &quokka_filler);
    try ids.appendSlice(a, &quokka_again);
    const n: u32 = @intCast(ids.items.len);
    const prefill = struct {
        fn run(m: *plugin.Loaded, al: std.mem.Allocator, cache: *test_cache.TestCache, tokens: []const u32, slot: u32) !u32 {
            var h = [_]abi.Handle{cache.handle()};
            var out: abi.ForwardOut = .{};
            try testing.expect(plugin.table.forward(m, &.{ .mode = .prefill, .logits = .last_row, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&[_]u32{slot}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
            return argmaxOf(try conformance.consumeF32(al, out.logits));
        }
    };
    var plain = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
    defer plain.deinit();
    var want: [48]u32 = undefined;
    var step_logits: [want.len][]f32 = undefined;
    want[0] = try prefill.run(rel.m, a, &plain, ids.items, 0);
    for (1..want.len) |i| {
        var h = [_]abi.Handle{plain.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(t.forward(rel.m, &.{ .mode = .decode, .logits = .last_row, .rows = 1, .tokens = .of(want[i - 1 .. i]), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{n + @as(u32, @intCast(i)) - 1}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
        step_logits[i] = try conformance.consumeF32(a, out.logits);
        want[i] = argmaxOf(step_logits[i]);
    }
    var cache = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
    defer cache.deinit();
    var t1 = try prefill.run(rel.m, a, &cache, ids.items, 1);
    var got: std.ArrayList(u32) = .empty;
    var pos: u32 = n;
    var verified: [abi.max_round_tokens]u32 = @splat(0);
    var kept: [abi.max_round_tokens]u32 = @splat(0);
    var first_draft: []const u32 = &.{};
    for (0..8) |r| {
        var h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(nd.verify.?(rel.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&[_]u32{t1}), .slots = .of(&[_]u32{1}), .positions = .of(&[_]u32{pos}), .caches = .of(&h), .cache_ops = &test_cache.ops, .accept_cap = 7 }, &out).isOk());
        const rows: usize = out.round.n;
        const logits = try conformance.consumeF32(a, out.logits);
        const vocab = logits.len / rows;
        var accepted: usize = 0;
        while (accepted + 1 < rows and argmaxOf(logits[accepted * vocab ..][0..vocab]) == out.round.tokens[accepted + 1]) accepted += 1;
        for (1..rows) |k| verified[k] += 1;
        for (1..accepted + 1) |k| kept[k] += 1;
        if (r == 0) first_draft = try a.dupe(u32, out.round.tokens[1..rows]);
        try got.appendSlice(a, out.round.tokens[0 .. 1 + accepted]);
        t1 = argmaxOf(logits[accepted * vocab ..][0..vocab]);
        var fin: abi.ForwardOut = .{};
        try testing.expect(nd.finish.?(rel.m, @intCast(accepted), &fin).isOk());
        pos = @intCast(fin.round.position);
    }
    std.debug.print("inkling2 release: DFlash2, a prompt of {d}: first draft {any}; draft positions 1 to {d} verified {any}, accepted {any}\n", .{ n, first_draft, d.cfg.block - 1, verified[1..d.cfg.block], kept[1..d.cfg.block] });
    try testing.expect(got.items.len > 8);
    const m = @min(got.items.len, want.len);
    var same: usize = 0;
    while (same < m and want[same] == got.items[same]) same += 1;
    if (same < m) {
        try testing.expect(same >= 1);
        const lg = step_logits[same];
        const gap = lg[want[same]] - lg[got.items[same]];
        std.debug.print("inkling2 release: DFlash2 rounds and plain decode part at token {d} ({d} vs {d}): the plain step's logit gap {d:.4}\n", .{ same, want[same], got.items[same], gap });
        try testing.expect(gap <= 0.01 * @abs(lg[want[same]]));
    }

    // A sampled round's verify draws its drafts from the selector's law and hands that law back, a row a draft.
    var sampled = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
    defer sampled.deinit();
    const s1 = try prefill.run(rel.m, a, &sampled, ids.items, 2);
    var h = [_]abi.Handle{sampled.handle()};
    var out: abi.ForwardOut = .{};
    try testing.expect(nd.verify.?(rel.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&[_]u32{s1}), .slots = .of(&[_]u32{2}), .positions = .of(&[_]u32{n}), .caches = .of(&h), .cache_ops = &test_cache.ops, .accept_cap = 7, .draft_temperature = 1.0, .draft_seed = 7 }, &out).isOk());
    const drafts = out.round.n - 1;
    a.free(try conformance.consumeF32(a, out.logits));
    try testing.expect(drafts > 0);
    const law = try conformance.consumeF32(a, out.draft_probs);
    const vocab = law.len / drafts;
    for (0..drafts) |i| {
        const row = law[i * vocab ..][0..vocab];
        var sum: f32 = 0;
        var support: u32 = 0;
        for (row) |x| {
            sum += x;
            support += @intFromBool(x > 0);
        }
        try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-3);
        try testing.expect(support <= d.cfg.top_k);
        try testing.expect(row[out.round.tokens[1 + i]] > 0);
    }
    var fin: abi.ForwardOut = .{};
    try testing.expect(nd.finish.?(rel.m, 0, &fin).isOk());
}

test "inkling2 proof: a step reads every visit's conv windows and each loop pass's own KV" {
    for (streams()) |gpu| try stateMutations(gpu);
}

fn stateMutations(gpu: bool) !void {
    const a = testing.allocator;
    var s = try Setup.init(a, gpu);
    defer s.deinit(a);
    const split = 12;
    const want = s.whole[split * VOCAB .. (split + 1) * VOCAB];
    const Mutation = enum { none, no_conv_state, passes_share_kv };
    for ([_]Mutation{ .none, .no_conv_state, .passes_share_kv }) |m| {
        var cache = try test_cache.TestCache.init(a, s.pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        a.free(try step(a, &s.pack, &cache, prompt[0..split], 0, .last_row));
        switch (m) {
            .none => {},
            .no_conv_state => for (cache.layers) |*l| {
                _ = mlx.mlx_array_free(l.conv);
                l.conv = .{};
            },
            // Visits 2..5 are the loop's first pass, 6..9 its second: the second pass given the first's keys.
            .passes_share_kv => for (2..6) |v| {
                try mlx.check(mlx.mlx_array_set(&cache.layers[v + 4].k, cache.layers[v].k));
                try mlx.check(mlx.mlx_array_set(&cache.layers[v + 4].v, cache.layers[v].v));
            },
        }
        const row = try step(a, &s.pack, &cache, prompt[split .. split + 1], split, .last_row);
        defer a.free(row);
        const cos = parity.qwen4CompareRows(row, want, 1, VOCAB).min_cos;
        if (m == .none) try testing.expect(cos > 0.999) else try testing.expect(cos < 0.99);
    }
}

// Bar: an image's rows reach the forward: the same prompt over two images' rows serves different last rows, and
// the same rows the same one.
test "inkling2 proof: media rows splice at the image placeholder" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const config = @import("config.zig");
    // The fixture with an image tower declared (its placeholder inside the tiny vocab) and none shipped.
    var fx = try plugin.conformanceFixture(arena.allocator());
    const files = try arena.allocator().dupe(@TypeOf(fx.files[0]), fx.files);
    files[0].bytes = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2,\"image_token_id\":59,\"vision_config\":{\"hidden_size\":32,\"num_hidden_layers\":1,\"num_attention_heads\":2}," ++ config.fixture_text ++ "}";
    fx.files = files;
    var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fx, .{ .gpu = false });
    defer pack.close();
    try testing.expect(pack.parsed.caps.images and pack.parsed.placeholder_ids.image == 59);
    const tokens = [_]u32{ 1, 17, 59, 59, 3 };
    var rows: [3][]f32 = undefined;
    for (&rows, [_]u64{ 5, 6, 5 }) |*r, seed| {
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        const data = try fixture.randomF32(a, seed, 2 * 64, 1.0);
        defer a.free(data);
        const media = mlx.mlx_array_new_data(data.ptr, &[_]c_int{ 1, 2, 64 }, 3, .float32);
        defer _ = mlx.mlx_array_free(media);
        var h = [_]abi.Handle{cache.handle()};
        var out: abi.ForwardOut = .{};
        const v = pack.table.forward(pack.model, &.{ .mode = .prefill, .logits = .last_row, .rows = tokens.len, .tokens = .of(&tokens), .slots = .of(&[_]u32{0}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops, .media = .{ .ctx = media.ctx } }, &out);
        try testing.expect(v.isOk());
        r.* = try conformance.consumeF32(a, out.logits);
    }
    defer for (rows) |r| a.free(r);
    try testing.expectEqualSlices(f32, rows[0], rows[2]);
    try testing.expect(parity.qwen4CompareRows(rows[0], rows[1], 1, VOCAB).min_cos < 0.999);
}

// Bar: a draft round's trunk half. A block of rows verified without appending, then its first rows committed, leaves
// the caches as a forward of those rows alone does: the next step serves the same row (cosine above 0.999), for
// every count kept; and before the commit the caches hold nothing of the block. (Rows against one-row steps is the
// CPU proof above: on the GPU a last bit flips a pick of this fixture.)
test "inkling2 proof: a verified block's kept rows leave the state a forward of those rows leaves" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
    const run_mod = @import("../../sdk/bases/run.zig");
    const visits = s.pack.parsed.shell.num_hidden_layers;
    const split = 9;
    const width = 6;
    for ([_]usize{ 1, 3, width }) |kept| {
        // The prompt, then the kept rows as one forward, then the step under test.
        var serial = try test_cache.TestCache.init(a, visits);
        defer serial.deinit();
        a.free(try step(a, &s.pack, &serial, prompt[0..split], 0, .last_row));
        a.free(try step(a, &s.pack, &serial, prompt[split .. split + kept], split, .last_row));
        const want = try step(a, &s.pack, &serial, prompt[split + kept .. split + kept + 1], split + kept, .last_row);
        defer a.free(want);

        var cache = try test_cache.TestCache.init(a, visits);
        defer cache.deinit();
        a.free(try step(a, &s.pack, &cache, prompt[0..split], 0, .last_row));
        var g = ops.Graph.init(a, m.base.s);
        defer g.deinit();
        const opens = try a.alloc(kernels.Open, visits);
        defer a.free(opens);
        @memset(opens, .{ .g = &g });
        var sp: plugin.Spec = .{ .visits = &.{}, .keep = 4 };
        defer {
            m.base.weights.spec = null;
            for ([_]mlx.mlx_array{ sp.ring, sp.held }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
        }
        m.base.weights.spec = &sp;
        sp.opens = opens;
        a.free(try step(a, &s.pack, &cache, prompt[split .. split + width], split, .all_rows));
        sp.opens = null;
        // Nothing of the block reached the caches.
        try testing.expectEqual(@as(u64, split), test_cache.ops.len(cache.handle(), 0));
        var h = [_]abi.Handle{cache.handle()};
        const pos = [_]u32{split};
        const r: run_mod.Run = .{ .mode = .verify, .rows = @intCast(kept), .positions = &pos, .caches = &h, .cache_ops = &test_cache.ops };
        var c = r.ctx(&g, &m.base.shared);
        try kernels.commit(&c, &m.base.weights.body.cfg, opens, @intCast(kept), 0, 1);
        try testing.expectEqual(@as(u64, split + kept), test_cache.ops.len(cache.handle(), 0));
        const got = try step(a, &s.pack, &cache, prompt[split + kept .. split + kept + 1], split + kept, .last_row);
        defer a.free(got);
        try testing.expect(parity.qwen4CompareRows(got, want, 1, VOCAB).min_cos > 0.999);
    }
}

/// One verify forward of `width` rows for each of several slots (`.batched_rows`): slot i's `tokens[i * width ..]` at
/// `positions[i]` on `caches[i]`; every row's logits.
fn groupStep(a: std.mem.Allocator, pack: *conformance.Loaded, caches: []test_cache.TestCache, tokens: []const u32, positions: []const u32, width: usize) ![]f32 {
    var handles: [8]abi.Handle = undefined;
    var slots: [8]u32 = undefined;
    for (caches, 0..) |*c, i| {
        handles[i] = c.handle();
        slots[i] = @intCast(i);
    }
    var out: abi.ForwardOut = .{};
    const v = pack.table.forward(pack.model, &.{ .mode = .batched_rows, .logits = .all_rows, .rows = @intCast(width), .tokens = .of(tokens), .slots = .of(slots[0..caches.len]), .positions = .of(positions), .caches = .of(handles[0..caches.len]), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

// Bar: the blocks of several slots verified as one forward read what each block's own verify reads (to the rows' cosine, at a
// width the packed expert kernels serve and one they do not), and kept to different lengths leave each slot's caches and conv
// states as a forward of its kept rows leaves them.
test "inkling2 proof: a verify of several slots' blocks serves what each slot's verify serves alone" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
    const run_mod = @import("../../sdk/bases/run.zig");
    const visits = s.pack.parsed.shell.num_hidden_layers;
    const splits = [_]usize{ 9, 12, 5 };
    const kept = [_]usize{ 3, 1, 4 };
    for ([_]usize{ 4, 6 }) |width| {
        var caches: [3]test_cache.TestCache = undefined;
        var alone: [3]test_cache.TestCache = undefined;
        for (&caches, &alone, splits) |*c, *l, split| {
            c.* = try test_cache.TestCache.init(a, visits);
            l.* = try test_cache.TestCache.init(a, visits);
            a.free(try step(a, &s.pack, c, prompt[0..split], 0, .last_row));
            a.free(try step(a, &s.pack, l, prompt[0..split], 0, .last_row));
        }
        defer for (&caches, &alone) |*c, *l| {
            c.deinit();
            l.deinit();
        };
        var g = ops.Graph.init(a, m.base.s);
        defer g.deinit();
        const opens = try a.alloc(kernels.Open, visits);
        defer a.free(opens);
        @memset(opens, .{ .g = &g });
        var sp: plugin.Spec = .{ .visits = &.{}, .keep = 4 };
        defer {
            m.base.weights.spec = null;
            for ([_]mlx.mlx_array{ sp.ring, sp.held }) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
        }
        m.base.weights.spec = &sp;
        var tokens: [3 * 6]u32 = undefined;
        var positions: [3]u32 = undefined;
        for (splits, 0..) |split, i| {
            @memcpy(tokens[i * width ..][0..width], prompt[split .. split + width]);
            positions[i] = @intCast(split);
        }
        sp.opens = opens;
        const rows = try groupStep(a, &s.pack, &caches, tokens[0 .. 3 * width], &positions, width);
        defer a.free(rows);
        sp.opens = null;
        for (splits, 0..) |split, i| {
            // Nothing of the block reached the cache, and the block reads what it reads alone.
            try testing.expectEqual(@as(u64, split), test_cache.ops.len(caches[i].handle(), 0));
            const opens_alone = try a.alloc(kernels.Open, visits);
            defer a.free(opens_alone);
            @memset(opens_alone, .{ .g = &g });
            sp.opens = opens_alone;
            const want_rows = try step(a, &s.pack, &alone[i], prompt[split .. split + width], split, .all_rows);
            defer a.free(want_rows);
            sp.opens = null;
            try testing.expect(parity.qwen4CompareRows(rows[i * width * VOCAB ..][0 .. width * VOCAB], want_rows, width, VOCAB).min_cos > 0.999);
        }
        for (splits, kept, 0..) |split, keep, i| {
            var serial = try test_cache.TestCache.init(a, visits);
            defer serial.deinit();
            a.free(try step(a, &s.pack, &serial, prompt[0..split], 0, .last_row));
            a.free(try step(a, &s.pack, &serial, prompt[split .. split + keep], split, .last_row));
            const want = try step(a, &s.pack, &serial, prompt[split + keep .. split + keep + 1], split + keep, .last_row);
            defer a.free(want);
            var h = [_]abi.Handle{caches[i].handle()};
            const pos = [_]u32{@intCast(split)};
            const r: run_mod.Run = .{ .mode = .verify, .rows = @intCast(keep), .positions = &pos, .caches = &h, .cache_ops = &test_cache.ops };
            var c = r.ctx(&g, &m.base.shared);
            try kernels.commit(&c, &m.base.weights.body.cfg, opens, @intCast(keep), i, 3);
            try testing.expectEqual(@as(u64, split + keep), test_cache.ops.len(caches[i].handle(), 0));
            const got = try step(a, &s.pack, &caches[i], prompt[split + keep .. split + keep + 1], split + keep, .last_row);
            defer a.free(got);
            try testing.expect(parity.qwen4CompareRows(got, want, 1, VOCAB).min_cos > 0.999);
        }
    }
}

/// One batched decode step: slot i's `tokens[i]` at `positions[i]` on `caches[i]`; the rows' logits.
fn batchedStep(a: std.mem.Allocator, pack: *conformance.Loaded, caches: []test_cache.TestCache, tokens: []const u32, positions: []const u32) ![]f32 {
    var handles: [8]abi.Handle = undefined;
    var slots: [8]u32 = undefined;
    for (caches, 0..) |*c, i| {
        handles[i] = c.handle();
        slots[i] = @intCast(i);
    }
    var out: abi.ForwardOut = .{};
    const v = pack.table.forward(pack.model, &.{ .mode = .batched_rows, .logits = .last_row, .rows = 1, .tokens = .of(tokens), .slots = .of(slots[0..caches.len]), .positions = .of(positions), .caches = .of(handles[0..caches.len]), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

// Bar: slots of different lengths step twice as batched forwards (the second reads the caches and conv states the
// first left per slot). CPU, a slot at a time: each row is what its slot serves alone, to the rows' bar. GPU, rows of
// the kernels: a slot's row does not move with its place or its neighbours, to 1e-5 of the top logit.
test "inkling2 proof: batched slots serve what each slot serves alone" {
    const a = testing.allocator;
    try batchedAlone(a);
    if (!(mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable())) try batchedApart(a);
}

/// Three slots, each a prompt prefix of its own length, prefilled on fresh caches.
const Slots = struct {
    caches: [3]test_cache.TestCache = undefined,
    made: usize = 0,

    fn init(s: *Slots, a: std.mem.Allocator, pack: *conformance.Loaded, lens: [3]usize) !void {
        for (lens, 0..) |n, i| {
            s.caches[i] = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
            s.made = i + 1;
            a.free(try step(a, pack, &s.caches[i], prompt[0..n], 0, .last_row));
        }
    }

    fn deinit(s: *Slots) void {
        for (s.caches[0..s.made]) |*c| c.deinit();
    }

    /// One batched step of every slot, slot i at `lens[i] + round`.
    fn next(s: *Slots, a: std.mem.Allocator, pack: *conformance.Loaded, lens: [3]usize, round: usize) ![]f32 {
        var tokens: [3]u32 = undefined;
        var positions: [3]u32 = undefined;
        for (lens, 0..) |n, i| {
            tokens[i] = prompt[n + round];
            positions[i] = @intCast(n + round);
        }
        return batchedStep(a, pack, &s.caches, &tokens, &positions);
    }
};

fn batchedAlone(a: std.mem.Allocator) !void {
    const lens = [_]usize{ 5, 9, 14 };
    var s = try Setup.init(a, false);
    defer s.deinit(a);
    var alone: Slots = .{};
    defer alone.deinit();
    var slots: Slots = .{};
    defer slots.deinit();
    try alone.init(a, &s.pack, lens);
    try slots.init(a, &s.pack, lens);
    for (0..2) |round| {
        var want: [lens.len * VOCAB]f32 = undefined;
        for (lens, 0..) |n, i| {
            const row = try step(a, &s.pack, &alone.caches[i], prompt[n + round ..][0..1], n + round, .last_row);
            defer a.free(row);
            @memcpy(want[i * VOCAB ..][0..VOCAB], row);
        }
        const got = try slots.next(a, &s.pack, lens, round);
        defer a.free(got);
        try expectRows(got, &want);
    }
}

fn batchedApart(a: std.mem.Allocator) !void {
    // A length is a sequence: the first run's slots, the same in another order, and its first slot beside two others.
    const runs = [_][3]usize{ .{ 5, 9, 14 }, .{ 14, 5, 9 }, .{ 7, 11, 5 } };
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    var slots: [runs.len]Slots = @splat(.{});
    defer for (&slots) |*x| x.deinit();
    for (&slots, runs) |*x, lens| try x.init(a, &s.pack, lens);
    for (0..2) |round| {
        var rows: [runs.len][]f32 = undefined;
        var made: usize = 0;
        defer for (rows[0..made]) |r| a.free(r);
        for (&slots, runs, 0..) |*x, lens, r| {
            rows[r] = try x.next(a, &s.pack, lens, round);
            made = r + 1;
        }
        for (runs[1..], rows[1..]) |lens, got| for (lens, 0..) |n, i| {
            const j = std.mem.indexOfScalar(usize, &runs[0], n) orelse continue;
            try expectSameRow(got[i * VOCAB ..][0..VOCAB], rows[0][j * VOCAB ..][0..VOCAB]);
        };
    }
}

fn expectSameRow(got: []const f32, want: []const f32) !void {
    var top: f32 = 0;
    var d: f32 = 0;
    for (got, want) |x, y| {
        top = @max(top, @abs(y));
        d = @max(d, @abs(x - y));
    }
    if (!(d <= 1e-5 * top)) std.debug.print("[inkling2 proof] batched row moved {e} of top {d:.4}\n", .{ d, top });
    try testing.expect(d <= 1e-5 * top);
}

// Bar: a sliding visit's cache is a ring of window + slack rows, declared to the engine; a ring short enough to drop
// rows serves, through the engine's own cache and past its first grow, the full buffer's logits bit for bit.
test "inkling2 proof: a sliding ring that drops rows serves what the full buffer serves" {
    const a = testing.allocator;
    const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
    const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
    const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
    const run_mod = @import("../../sdk/bases/run.zig");
    const sliding_ring = @import("../../sdk/config/sliding_ring.zig");
    for (streams()) |gpu| {
        var s = try Setup.init(a, gpu);
        defer s.deinit(a);
        const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
        const cfg = &m.base.weights.body.cfg;
        try testing.expectEqual(sliding_ring.rows(cfg.sliding_window), cfg.ring);
        try testing.expectEqual(cfg.ring, s.pack.parsed.cache.sliding_ring);
        var full = m.base.weights;
        full.body.cfg.ring = 0;
        var ringed = m.base.weights;
        ringed.body.cfg.ring = cfg.sliding_window + 2;
        const visits = s.pack.parsed.shell.num_hidden_layers;
        var kvs: [2]KVCache = undefined;
        var entries: [2][]SSMCacheEntry = undefined;
        var slots: [2]cache_ops.Slot = undefined;
        var made: usize = 0;
        defer for (kvs[0..made], entries[0..made]) |*c, es| {
            c.deinit();
            for (es) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            a.free(es);
        };
        for (&kvs, &entries, &slots) |*c, *es, *sl| {
            c.* = try KVCache.init(a, visits);
            es.* = try a.alloc(SSMCacheEntry, visits);
            for (es.*) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            made += 1;
            sl.* = .{ .kv = c, .ssm = es.* };
        }
        // A prompt, a step, a chunk past the cache's first capacity, steps, a chunk past the next (a grow that drops
        // ring rows, in whole blocks), steps.
        var ids: [620]i32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast((i * 37 + 11) % VOCAB);
        const steps = [_][2]usize{ .{ 0, 10 }, .{ 10, 11 }, .{ 11, 311 }, .{ 311, 312 }, .{ 312, 313 }, .{ 313, 613 }, .{ 613, 614 }, .{ 614, 615 } };
        for (steps) |st| {
            var g = ops.Graph.init(a, m.base.s);
            defer g.deinit();
            const n: c_int = @intCast(st[1] - st[0]);
            const tok = try g.fromSlice(i32, ids[st[0]..st[1]], &.{ 1, n });
            var outs: [2]ops.Array = undefined;
            for ([_]*const plugin.Model.Weights{ &full, &ringed }, &slots, 0..) |w, *sl, arm| {
                const h = [_]abi.Handle{sl.handle()};
                const pos = [_]u32{@intCast(st[0])};
                const r: run_mod.Run = .{ .mode = if (n == 1) .decode else .prefill, .rows = n, .positions = &pos, .caches = &h, .cache_ops = &cache_ops.ops, .logits = .all_rows };
                outs[arm] = (try plugin.Model.forward(w, &m.base.shared, &g, tok, r)).logits;
            }
            try ops.expectBitEqual(outs[1], outs[0]);
        }
        for (kvs, 0..) |c, arm| {
            var dropped = false;
            for (c.entries) |e| dropped = dropped or e.base > 0;
            try testing.expectEqual(arm == 1, dropped);
        }
    }
}

// Bar: the batched slots' one-launch read against the per-slot read it replaces (a mask and MLX's attention a slot),
// at the same slots: 2, 8 and 10 (two launches) slots of their own lengths, each a view of the first keys of a longer
// buffer as a cache holds them, on a sliding and a global layer, at the fixture's 16-wide heads and the model's 128.
// f32 1e-4, bf16 2e-2 of unit-scale values.
test "inkling2 kernels: batched slots read in one launch as the per-slot read does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const Geo = struct { heads: c_int, kv_heads: c_int, hd: c_int, extent: c_int, window: c_int };
    var seed: u64 = 16000;
    for ([_]Geo{ .{ .heads = 4, .kv_heads = 2, .hd = 16, .extent = 8, .window = 6 }, .{ .heads = 24, .kv_heads = 8, .hd = 128, .extent = 1024, .window = 1024 } }) |geo| for ([_]usize{ 2, 8, 10 }) |n| for ([_]bool{ true, false }) |sliding| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const slots: c_int = @intCast(n);
        var ks: [10]ops.Array = undefined;
        var vs: [10]ops.Array = undefined;
        for (0..n) |i| {
            const keys: c_int = @intCast(3 + 157 * i);
            const kv = [2]ops.Array{ try randomRow(&g, a, seed + 2 * i, &.{ 1, geo.kv_heads, keys + 24, geo.hd }, 0.5, dt), try randomRow(&g, a, seed + 2 * i + 1, &.{ 1, geo.kv_heads, keys + 24, geo.hd }, 1.0, dt) };
            ks[i] = try g.sliceAxis(kv[0], 2, 0, keys);
            vs[i] = try g.sliceAxis(kv[1], 2, 0, keys);
        }
        const q = try randomRow(&g, a, seed + 100, &.{ 1, slots, geo.heads, geo.hd }, 0.5, dt);
        const bias = try randomRow(&g, a, seed + 101, &.{ 1, slots, geo.heads, geo.extent }, 1.0, dt);
        const sinks = try randomRow(&g, a, seed + 102, &.{ 1, slots, geo.heads }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 103, &.{slots * geo.heads * geo.hd}, 2.0, .float32));
        const window: c_int = if (sliding) geo.window else 0;
        const want = try kernels.slotsChain(&g, try g.transpose(q, &.{ 0, 2, 1, 3 }), ks[0..n], vs[0..n], bias, sinks, window, gate);
        const got = try kernels.slotsRead(&g, q, ks[0..n], vs[0..n], bias, try g.contiguous(try g.astype(sinks, .float32)), window, gate);
        try expectClose(got, want, if (dt == .float32) 1e-4 else 2e-2, "batched slots' read");
    };
}

const ops = @import("../../mlx/ops.zig");
const layers = @import("layers.zig");
const kernels = @import("kernels.zig");
fn randomRow(g: *ops.Graph, a: std.mem.Allocator, seed: u64, shape: []const c_int, scale: f32, dt: ops.Dtype) !ops.Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try fixture.randomF32(a, seed, n, scale);
    defer a.free(data);
    return g.astype(try g.fromSlice(f32, data, shape), dt);
}

fn expectClose(got: ops.Array, want: ops.Array, tol: f32, what: []const u8) !void {
    try testing.expectEqual(mlx.mlx_array_size(want), mlx.mlx_array_size(got));
    const d = parity.maxAbsDiffF32(got, want, fx_stream) catch |e| {
        std.debug.print("[inkling2 kernels] {s}: {s}\n", .{ what, @errorName(e) });
        return e;
    };
    if (!(d <= tol)) std.debug.print("[inkling2 kernels] {s}: max diff {d:.6} over {d:.6}\n", .{ what, d, tol });
    try testing.expect(d <= tol);
}

var fx_stream: mlx.mlx_stream = undefined;

/// The fixture pack on the GPU's stream, for a stage's kernels against its op chain.
const Stage = struct {
    arena: std.heap.ArenaAllocator,
    pack: conformance.Loaded,
    body: *const layers.Body,

    fn init(a: std.mem.Allocator) !Stage {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena.allocator()), .{ .gpu = true });
        const m: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
        fx_stream = m.base.s;
        return .{ .arena = arena, .pack = pack, .body = &m.base.weights.body };
    }

    fn deinit(s: *Stage) void {
        s.pack.close();
        s.arena.deinit();
    }
};

// Layer 0 opens the first block and reads its own input; 1 is global inside it; 2 opens one in the loop; 3 rides it
// and 5 is the loop's global layer.
const StageCase = struct { layer: usize, blocks: c_int, messages: bool };
const stage_cases = [_]StageCase{
    .{ .layer = 0, .blocks = 0, .messages = false }, .{ .layer = 1, .blocks = 1, .messages = false },
    .{ .layer = 2, .blocks = 1, .messages = false }, .{ .layer = 2, .blocks = 1, .messages = true },
    .{ .layer = 3, .blocks = 2, .messages = true },  .{ .layer = 5, .blocks = 3, .messages = true },
};

// Bar: each stage's kernels against the op chain they stand for, on the same inputs, at one row and at a prompt's
// rows. f32: 1e-4, and 2e-2 where the chain goes through the GPU's f32 matmul (good to about 1e-3 per product). bf16:
// a bf16 step of the largest values (0.05 to 0.1), since a last-bit difference rounds an element the other way.
test "inkling2 kernels: the stream kernels serve the projections of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const hidden: c_int = @intCast(s.body.cfg.hidden_size);
    var seed: u64 = 7000;
    for (stage_cases) |case| for ([_]c_int{ 1, 7 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &s.body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const h = try randomRow(&g, a, seed, &.{ 1, rows, hidden }, 1.0, dt);
        const blocks: ?ops.Array = if (case.blocks > 0) try randomRow(&g, a, seed + 1, &.{ 1, rows, case.blocks, hidden }, 1.0, .float32) else null;
        const bank: ?[2]ops.Array = if (case.messages) .{ try randomRow(&g, a, seed + 2, &.{ 1, rows, 2, hidden }, 1.0, .float32), try randomRow(&g, a, seed + 3, &.{ 1, rows, hidden }, 1.0, .float32) } else null;
        try testing.expect(kernels.runs(&g, L, &s.body.cfg, h));
        const got = try kernels.pre(&g, L, s.body, h, blocks, bank);
        const messages: ?ops.Array = if (bank) |bk| try layers.messagesFor(&g, L, bk[0], bk[1]) else null;
        const want = try layers.project(&g, L, s.body, h, blocks, messages);
        const tol: f32 = if (dt == .float32) 2e-2 else 0.1;
        for (got.proj, want.proj) |x, y| try expectClose(x, y, tol, "projection");
        try testing.expectEqual(want.dyn != null, got.dyn != null);
        if (want.dyn) |d| for (got.dyn.?, d) |x, y| try expectClose(x, y, tol, "conv taps");
        try expectClose(got.normed, want.normed, 1e-4, "normed stream");
        // The chain's messages are `[1, S, 4, D]`.
        if (messages) |m| try expectClose(got.messages.?, try g.contiguous(try g.transpose(m, &.{ 2, 1, 0, 3 })), 2e-2, "messages");
    };
    var g = ops.Graph.init(a, fx_stream);
    defer g.deinit();
    const h = try randomRow(&g, a, seed + 20, &.{ 1, 3, hidden }, 1.0, .bfloat16);
    const blocks = try randomRow(&g, a, seed + 21, &.{ 1, 3, 3, hidden }, 1.0, .float32);
    const nw = try randomRow(&g, a, seed + 22, &.{hidden}, 1.0, .float32);
    const got = (try kernels.finalStream(&g, s.body, h, blocks, nw)) orelse return error.TestUnexpectedResult;
    try expectClose(got, try g.astype(try g.rmsNorm(try layers.finalInput(&g, s.body, h, blocks), nw, s.body.cfg.rms_norm_eps), .bfloat16), 0.05, "final stream");
}

// Bar: as above. A 32-thread group (the narrowest an M1 falls to) with more sums than threads, on a looped global layer.
test "inkling2 kernels: the stream kernel at its narrowest group folds every sum" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const saved = kernels.stream_threads;
    defer kernels.stream_threads = saved;
    kernels.stream_threads = 32;
    const hidden: c_int = @intCast(s.body.cfg.hidden_size);
    const L = &s.body.layers[5];
    for ([_]c_int{ 1, 7 }) |rows| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const h = try randomRow(&g, a, 9100, &.{ 1, rows, hidden }, 1.0, .float32);
        // 1 + 3 queries x 7 sources + 4 messages x 4 sources = 38 sums.
        const blocks = try randomRow(&g, a, 9101, &.{ 1, rows, 6, hidden }, 1.0, .float32);
        const bank: [2]ops.Array = .{ try randomRow(&g, a, 9102, &.{ 1, rows, 3, hidden }, 1.0, .float32), try randomRow(&g, a, 9103, &.{ 1, rows, hidden }, 1.0, .float32) };
        const got = try kernels.pre(&g, L, s.body, h, blocks, bank);
        const want = try layers.project(&g, L, s.body, h, blocks, try layers.messagesFor(&g, L, bank[0], bank[1]));
        for (got.proj, want.proj) |x, y| try expectClose(x, y, 2e-2, "projection");
        try expectClose(got.normed, want.normed, 1e-4, "normed stream");
    }
}

/// Random projections of `rows` rows for a layer, as `layers.project` hands them over.
fn randomProjected(g: *ops.Graph, a: std.mem.Allocator, seed: u64, L: *const layers.Layer, cfg: *const @TypeOf(@as(layers.Body, undefined).cfg), rows: c_int, dt: ops.Dtype) !layers.Projected {
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const kv_dim: c_int = @intCast(cfg.kvDim());
    const taps: c_int = @intCast(cfg.num_key_value_heads * cfg.sconv_kernel);
    var proj: layers.Projected = undefined;
    for (&proj.proj, [_]c_int{ heads * hd, heads * hd, heads * @as(c_int, @intCast(cfg.d_rel)), heads, kv_dim, kv_dim }, 0..) |*x, n, i|
        x.* = try randomRow(g, a, seed + i, &.{ 1, rows, n }, 1.5, dt);
    proj.dyn = if (L.global) .{ try randomRow(g, a, seed + 50, &.{ 1, rows, taps }, 0.3, dt), try randomRow(g, a, seed + 51, &.{ 1, rows, taps }, 0.3, dt) } else null;
    proj.stream = proj.proj[0];
    proj.normed = proj.proj[0];
    return proj;
}

// Bar: as above, on a sliding and a global layer. One row: with and without log-scaling, its bias row for a view
// shorter than, as long as and longer than the relative extent. A prompt's rows: the clipped relative states.
test "inkling2 kernels: the prep kernel serves the values of the chain it replaces" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const width: c_int = @intCast(2 * cfg.kvDim() + 2 * cfg.hidden_size);
    var seed: u64 = 900;
    for ([_]usize{ 0, 1 }) |li| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &body.layers[li];
        const extent = ops.shape(L.rel_proj)[1];
        const tol: f32 = if (dt == .float32) 1e-4 else 0.05;
        for ([_]f32{ 1.0, 1.25 }) |tau| for ([_]c_int{ 3, 0, 5 }) |past| {
            const keys = if (past == 3) 3 else extent + past;
            var g = ops.Graph.init(a, fx_stream);
            defer g.deinit();
            seed += 100;
            const proj = try randomProjected(&g, a, seed, L, cfg, 1, dt);
            const st = try randomRow(&g, a, seed + 60, &.{ 1, taps - 1, width }, 1.0, .float32);
            const got = (try kernels.prep(&g, L, cfg, &proj.proj, proj.dyn, st, tau, keys)) orelse return error.TestUnexpectedResult;
            const want = try layers.prepare(&g, L, body, &proj, st);
            try expectClose(got.q, try g.astype(try g.mulScalar(try g.astype(want.q, .float32), tau), dt), tol, "q");
            try expectClose(got.k, want.k, tol, "k");
            try expectClose(got.v, want.v, tol, "v");
            try expectClose(got.gate, want.gate, tol, "gate");
            // The chain's one-row bias: the logits' tail, zero past the extent.
            var bias = if (keys < extent) try g.sliceAxis(want.rel, 1, extent - keys, extent) else want.rel;
            if (keys > extent) bias = try g.padAxis(bias, 1, keys - extent, 0);
            try expectClose(got.bias, try g.astype(try g.mulScalar(bias, tau), dt), @max(tol, 2e-2), "bias");
        };
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 100;
        const proj = try randomProjected(&g, a, seed, L, cfg, 7, dt);
        const st = try randomRow(&g, a, seed + 60, &.{ 1, taps - 1, width }, 1.0, .float32);
        const got = (try kernels.prep(&g, L, cfg, &proj.proj, proj.dyn, st, 1.0, 0)) orelse return error.TestUnexpectedResult;
        const want = try layers.prepare(&g, L, body, &proj, st);
        try expectClose(got.q, want.q, tol, "rows' q");
        // The chain's k and v are transposed views.
        try expectClose(got.k, try g.contiguous(want.k), tol, "rows' k");
        try expectClose(got.v, try g.contiguous(want.v), tol, "rows' v");
        try expectClose(got.gate, want.gate, tol, "rows' gate");
        try expectClose(got.rel, want.rel, tol, "rows' relative states");
    };
}

// Bar: at its narrowest group (32 threads) the mid kernel folds every one of its sums: forty open blocks give it more
// sums than threads, plain and on a looped pass; the chain's values as above (f32, 1e-4; the MLP input 2e-2).
test "inkling2 kernels: at its narrowest group the mid kernel folds every sum" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const q_dim: c_int = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const saved = kernels.stream_threads;
    defer kernels.stream_threads = saved;
    kernels.stream_threads = 32;
    var seed: u64 = 16000;
    for ([_]StageCase{ .{ .layer = 1, .blocks = 40, .messages = false }, .{ .layer = 3, .blocks = 40, .messages = true } }) |case| {
        const L = &body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const attn = try randomRow(&g, a, seed, &.{ 1, 1, q_dim }, 1.0, .float32);
        const h = try randomRow(&g, a, seed + 1, &.{ 1, 1, hidden }, 1.0, .float32);
        const normed = try randomRow(&g, a, seed + 2, &.{ 1, 1, hidden }, 1.0, .float32);
        const blocks = try randomRow(&g, a, seed + 3, &.{ 1, 1, case.blocks, hidden }, 1.0, .float32);
        const messages: ?ops.Array = if (case.messages) try randomRow(&g, a, seed + 4, &.{ 4, 1, hidden }, 1.0, .float32) else null;
        const st = try randomRow(&g, a, seed + 5, &.{ 1, taps - 1, 2 * @as(c_int, @intCast(cfg.kvDim())) + 2 * hidden }, 1.0, .float32);
        const got = try kernels.mid(&g, L, body, attn, null, h, normed, blocks, st, messages);
        const want = try layers.mid(&g, L, body, attn, h, normed, blocks, st, if (messages) |m| try g.contiguous(try g.transpose(try g.reshape(m, &.{ 4, 1, 1, hidden }), &.{ 2, 1, 0, 3 })) else null);
        try expectClose(got.residual, want.residual, 1e-4, "residual at 32 threads");
        try expectClose(got.x, want.x, 2e-2, "mlp input at 32 threads");
    }
}

// Bar: as above; the picks are the chain's experts, each at its chain weight.
test "inkling2 kernels: the mid kernels serve the values of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const top_k: usize = cfg.top_k;
    const q_dim: c_int = @intCast(cfg.num_attention_heads * cfg.head_dim);
    var seed: u64 = 5000;
    for (stage_cases) |case| for ([_]c_int{ 1, 7 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const attn = try randomRow(&g, a, seed, &.{ 1, rows, q_dim }, 1.0, dt);
        const h = try randomRow(&g, a, seed + 1, &.{ 1, rows, hidden }, 1.0, dt);
        const normed = try randomRow(&g, a, seed + 2, &.{ 1, rows, hidden }, 1.0, .float32);
        const blocks: ?ops.Array = if (case.blocks > 0) try randomRow(&g, a, seed + 3, &.{ 1, rows, case.blocks, hidden }, 1.0, .float32) else null;
        const messages: ?ops.Array = if (case.messages) try randomRow(&g, a, seed + 4, &.{ 4, rows, hidden }, 1.0, .float32) else null;
        const st = try randomRow(&g, a, seed + 5, &.{ 1, taps - 1, 2 * @as(c_int, @intCast(cfg.kvDim())) + 2 * hidden }, 1.0, .float32);
        const got = try kernels.mid(&g, L, body, attn, null, h, normed, blocks, st, messages);
        // The chain's messages are `[1, S, 4, D]`.
        const want = try layers.mid(&g, L, body, attn, try g.astype(h, .float32), normed, blocks, st, if (messages) |m| try g.contiguous(try g.transpose(try g.reshape(m, &.{ 4, rows, 1, hidden }), &.{ 2, 1, 0, 3 })) else null);
        const tol: f32 = if (dt == .float32) 1e-4 else 0.1;
        try expectClose(got.residual, want.residual, tol, "residual");
        try expectClose(got.x, want.x, @max(tol, 2e-2), "mlp input");
        try testing.expectEqual(want.blocks != null, got.blocks != null);
        if (want.blocks) |b| try expectClose(got.blocks.?, b, 1e-6, "block stack");
        const picks = try kernels.topK(&g, got.scores, L.router_bias, try g.fromSlice(f32, &.{cfg.route_scale}, &.{1}), @intCast(top_k));
        const gi = try hostValues(a, &g, picks[0]);
        defer a.free(gi);
        const wi = try hostValues(a, &g, want.idx);
        defer a.free(wi);
        const gw = try hostValues(a, &g, picks[1]);
        defer a.free(gw);
        const ww = try hostValues(a, &g, want.weights);
        defer a.free(ww);
        for (gi, gw, 0..) |e, weight, at| {
            const row = at / top_k * top_k;
            const pick = std.mem.indexOfScalar(f32, wi[row .. row + top_k], e) orelse {
                std.debug.print("[inkling2 kernels] layer {d}: expert {d} is not a chain pick {any}\n", .{ case.layer, e, wi[row .. row + top_k] });
                return error.TestUnexpectedResult;
            };
            try testing.expectApproxEqAbs(ww[row + pick], weight, 2e-2 * cfg.route_scale);
        }
    };
}

// Bar: wide forwards (`S >= 64`) use the 1-TG-per-row stream kernels (including multi-element-per-thread `D/TG > 1`)
// and route looped visits via GEMM (`s > packed_rows`), matching the chain `layers.pre` and `layers.mid`.
test "inkling2 kernels: wide prefill 1-TG stream kernels and looped router GEMM match the chain" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const body = s.body;
    const cfg = &body.cfg;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const taps: c_int = @intCast(cfg.sconv_kernel);
    const q_dim: c_int = @intCast(cfg.num_attention_heads * cfg.head_dim);
    const saved_tg = kernels.stream_threads;
    defer kernels.stream_threads = saved_tg;
    // Fixture hidden=64, so stream_threads=32 exercises D/TG = 2 in ONE_TG=1 mode.
    kernels.stream_threads = 32;
    const rows: c_int = kernels.packed_rows + 1;
    var seed: u64 = 5700;
    for (stage_cases) |case| {
        const L = &body.layers[case.layer];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const h = try randomRow(&g, a, seed, &.{ 1, rows, hidden }, 1.0, .bfloat16);
        const blocks: ?ops.Array = if (case.blocks > 0) try randomRow(&g, a, seed + 1, &.{ 1, rows, case.blocks, hidden }, 1.0, .float32) else null;
        const bank_arr: ?[2]ops.Array = if (case.messages) .{
            try randomRow(&g, a, seed + 2, &.{ 1, rows, 3, hidden }, 1.0, .float32),
            try randomRow(&g, a, seed + 3, &.{ 1, rows, hidden }, 1.0, .float32),
        } else null;
        const got_pre = try kernels.pre(&g, L, body, h, blocks, bank_arr);
        const msgs_want: ?ops.Array = if (bank_arr) |bk| try layers.messagesFor(&g, L, bk[0], bk[1]) else null;
        const want_pre = try layers.project(&g, L, body, h, blocks, msgs_want);
        try expectClose(got_pre.normed, want_pre.normed, 1e-4, "wide pre normed");
        try expectClose(got_pre.proj[0], want_pre.proj[0], 5e-2, "wide pre q_proj");

        const attn = try randomRow(&g, a, seed + 4, &.{ 1, rows, q_dim }, 1.0, .bfloat16);
        const st = try randomRow(&g, a, seed + 5, &.{ 1, taps - 1, 2 * @as(c_int, @intCast(cfg.kvDim())) + 2 * hidden }, 1.0, .float32);
        const got_mid = try kernels.mid(&g, L, body, attn, null, h, got_pre.normed, blocks, st, got_pre.messages);
        const want_mid = try layers.mid(&g, L, body, attn, try g.astype(h, .float32), want_pre.normed, blocks, st, msgs_want);
        try expectClose(got_mid.residual, want_mid.residual, 0.1, "wide mid residual");
        try expectClose(got_mid.x, want_mid.x, 0.1, "wide mid x");
        const want_scores = try g.matmul(try g.astype(got_mid.x, .float32), try g.transpose(L.router, &.{ 1, 0 }));
        try expectClose(got_mid.scores, want_scores, 1e-5, "wide looped router scores");
    }
}

fn hostValues(a: std.mem.Allocator, g: *ops.Graph, x: ops.Array) ![]f32 {
    return conformance.consumeF32(a, .{ .ctx = g.escape(try g.astype(try g.contiguous(x), .float32)).ctx });
}

const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const device = @import("../../mlx/device.zig");

/// A random expert bank `shape` (`[E, out, in]`) quantized by MLX into a packed layout.
fn packedBank(g: *ops.Graph, a: std.mem.Allocator, seed: u64, shape: []const c_int, bits: u32, group: u32, mode: QuantMode) !@TypeOf(@as(layers.Layer, undefined).gate_up) {
    const w = try randomRow(g, a, seed, shape, 0.1, .float32);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, w, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{}, g.s));
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, parts, 0));
    const words = try g.adopt(q);
    var sc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&sc, parts, 1));
    var bank: @TypeOf(@as(layers.Layer, undefined).gate_up) = .{ .w = words, .s = try g.adopt(sc), .qp = .{ .bits = bits, .group_size = group, .mode = mode } };
    if (mode == .affine) {
        var bi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&bi, parts, 2));
        bank.b = try g.adopt(bi);
    }
    return bank;
}

/// The affine widths an expert bank may be re-quantized to (`tests/convert_inkling2.py --experts`).
const affine_bits = [_]u32{ 2, 3, 4, 5, 6, 8 };
const affine_groups = [_]u32{ 32, 64, 128 };
// Bar: one row's fused output projection serves `projectRows(gated(read, gate))` within one bf16 rounding, on every
// affine width and group a pack's attention may carry and on a dense weight.
test "inkling2 kernels: the output projection kernel serves the gated read through the projection" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const n: c_int = 256;
    const heads: c_int = 24;
    const hd: c_int = 128;
    var seed: u64 = 9100;
    for ([_]?u32{ null, 2, 3, 4, 5, 6, 8 }) |bits| for (affine_groups) |group| {
        if (bits == null and group != 64) continue;
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        var o: layers.Linear = .{ .w = try randomRow(&g, a, seed, &.{ n, heads * hd }, 0.02, .bfloat16) };
        if (bits) |b| {
            const q = try packedBank(&g, a, seed, &.{ 1, n, heads * hd }, b, group, .affine);
            o = .{ .w = try g.reshape(q.w, &.{ n, -1 }), .s = try g.reshape(q.s, &.{ n, -1 }), .b = try g.reshape(q.b, &.{ n, -1 }), .bits = b, .group_size = group };
        }
        try testing.expect(kernels.oFormat(&o, .bfloat16) != null);
        const att = try randomRow(&g, a, seed + 1, &.{ 1, heads, 1, hd }, 1.0, .bfloat16);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 2, &.{heads * hd}, 2.0, .float32));
        const want = try layers.projectRows(&g, &o, try kernels.gated(&g, att, gate, 1));
        try expectClose(try kernels.oGated(&g, &o, att, gate), want, 4e-2, "output projection");
    };
}

const ExpertFormat = struct { mode: ?QuantMode, bits: u32 = 0, group: u32 = 0 };
/// Each width at the fixture's 64-wide stream: two scale groups of 32.
const affine_formats = blk: {
    var all: [affine_bits.len]ExpertFormat = undefined;
    for (affine_bits, 0..) |bits, i| all[i] = .{ .mode = .affine, .bits = bits, .group = 32 };
    break :blk all;
};

// Bar: the tail's launches against `layers.tail` from the router's own picks, with the new conv state (the old rows
// behind the rows' k, v and output projections and the MLP's output): on the fixture's dense bank and on each packed
// and affine layout a bank can load in, with and without per-expert scales, at one row (the packed banks read in place), a few
// rows (unsorted pairs) and a prompt's (sorted). f32 2e-3 (MLX's fp kernels sum in another order), bf16 0.3 (two steps of its largest sums).
test "inkling2 kernels: the expert tail kernels serve the stream and the conv state of the chain they replace" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    var body = s.body.*;
    const hidden: c_int = @intCast(body.cfg.hidden_size);
    const experts: c_int = @intCast(body.cfg.num_experts);
    const kv_dim: c_int = @intCast(body.cfg.kvDim());
    const taps: c_int = @intCast(body.cfg.sconv_kernel);
    var seed: u64 = 9000;
    for ([_]ExpertFormat{ .{ .mode = null }, .{ .mode = .nvfp4, .bits = 4, .group = 16 }, .{ .mode = .mxfp4, .bits = 4, .group = 32 }, .{ .mode = .mxfp8, .bits = 8, .group = 32 } } ++ affine_formats) |f| for ([_]c_int{ 1, 2, 7, 15, 16, 43 }) |rows| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| for ([_]bool{ true, false }) |scaled| for ([_]c_int{ 64, 1152 }) |inter_w| {
        // MLX's fp matmuls take bf16 rows; one row reads a packed bank in place at either.
        if (f.mode != null and dt == .float32 and rows > 1) continue;
        // 1152 is the release's width (4.5 sweeps of the down kernel), run for its format.
        if (inter_w == 1152 and (f.mode != .nvfp4 or dt == .float32)) continue;
        if (f.mode == null and inter_w != 64) continue;
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        var L = body.layers[3];
        body.cfg.moe_intermediate_size = s.body.cfg.moe_intermediate_size;
        if (f.mode) |mode| {
            // 64 is two scale groups wide: MLX's sorted MXFP8 gather is not finite on a bank one group (32) wide.
            body.cfg.moe_intermediate_size = @intCast(inter_w);
            L.gate_up = try packedBank(&g, a, seed, &.{ experts, 2 * inter_w, hidden }, f.bits, f.group, mode);
            L.down = try packedBank(&g, a, seed + 1, &.{ experts, hidden, inter_w }, f.bits, f.group, mode);
        }
        if (scaled) {
            L.gate_up_scale = try g.addScalar(try randomRow(&g, a, seed + 2, &.{experts}, 0.3, .float32), 1.5);
            L.down_scale = try g.addScalar(try randomRow(&g, a, seed + 3, &.{experts}, 0.3, .float32), 1.5);
        }
        const cfg = &body.cfg;
        const x = try randomRow(&g, a, seed + 4, &.{ 1, rows, hidden }, 1.0, dt);
        const r = try layers.route(&g, &L, cfg, x);
        const residual = try randomRow(&g, a, seed + 5, &.{ 1, rows, hidden }, 1.0, .float32);
        const st = try randomRow(&g, a, seed + 6, &.{ 1, taps - 1, 2 * kv_dim + 2 * hidden }, 1.0, .float32);
        const kp = try randomRow(&g, a, seed + 7, &.{ 1, rows, kv_dim }, 1.0, dt);
        const vp = try randomRow(&g, a, seed + 8, &.{ 1, rows, kv_dim }, 1.0, dt);
        const op = try randomRow(&g, a, seed + 9, &.{ 1, rows, hidden }, 1.0, dt);
        const scores = try g.matmul(try g.astype(x, .float32), try g.transpose(L.router, &.{ 1, 0 }));
        const got = try kernels.experts(&g, &L, cfg, &.{ .x = x, .scores = scores, .residual = residual, .blocks = null, .op = op }, kp, vp, st, null);
        const want = try layers.tail(&g, &L, &body, &.{ .x = x, .idx = r.idx, .weights = r.weights, .residual = residual, .blocks = null, .attn_state = x }, st);
        // The bf16 sums reach 20 and over, where one bf16 step is 0.125: a last-bit flip or two is not a miss.
        const tol: f32 = if (dt == .float32) 2e-3 else 0.3;
        var label: [96]u8 = undefined;
        try expectClose(got.h, want.h, tol, try std.fmt.bufPrint(&label, "tail stream ({s}, {d} rows, {s}, scaled {})", .{ if (f.mode) |m| @tagName(m) else "dense", rows, @tagName(dt), scaled }));
        // The k, v and attention windows: the last rows of the old state behind the rows' projections.
        const fed = try g.concat(&.{ try g.sliceAxis(st, 2, 0, 2 * kv_dim + hidden), try g.astype(try g.concat(&.{ kp, vp, op }, 2), .float32) }, 1);
        const left = try g.sliceAxis(fed, 1, rows, rows + taps - 1);
        try expectClose(got.state, try g.concat(&.{ left, want.mlp_state }, 2), tol, "conv state");
    };
}

// Bar: the gate_up launch that ranks the router's logits itself writes the top-k kernel's picks and weights bit for
// bit (a tied pair and a NaN logit among them), and the launch given those picks the same rows, on every packed
// layout, one row and a few, with and without scales.
test "inkling2 kernels: the gate_up launch ranks the router's picks as the top-k kernel does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    var cfg = s.body.cfg;
    cfg.num_experts = 32;
    cfg.top_k = 8;
    cfg.moe_intermediate_size = 64;
    const hidden: c_int = @intCast(cfg.hidden_size);
    var seed: u64 = 15000;
    for ([_]ExpertFormat{ .{ .mode = .nvfp4, .bits = 4, .group = 16 }, .{ .mode = .mxfp4, .bits = 4, .group = 32 }, .{ .mode = .mxfp8, .bits = 8, .group = 32 } } ++ affine_formats) |f| for ([_]c_int{ 1, 3 }) |rows| for ([_]bool{ true, false }) |scaled| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        var L = s.body.layers[3];
        L.gate_up = try packedBank(&g, a, seed, &.{ 32, 128, hidden }, f.bits, f.group, f.mode.?);
        L.router_bias = try randomRow(&g, a, seed + 1, &.{32}, 0.05, .float32);
        L.gate_up_scale = if (scaled) try g.addScalar(try randomRow(&g, a, seed + 2, &.{32}, 0.3, .float32), 1.5) else .{};
        const x = try randomRow(&g, a, seed + 3, &.{ 1, rows, hidden }, 1.0, .bfloat16);
        const logits = try fixture.randomF32(a, seed + 4, @intCast(rows * 32), 3.0);
        defer a.free(logits);
        logits[5] = logits[9];
        logits[12] = std.math.nan(f32);
        const scores = try g.fromSlice(f32, logits, &.{ rows, 32 });
        const want = try kernels.topK(&g, scores, L.router_bias, try g.fromSlice(f32, &.{cfg.route_scale}, &.{1}), 8);
        const got = try kernels.ups(&g, &L, &cfg, x, .{ .scores = scores });
        try expectClose(try g.astype(got[1], .float32), try g.astype(want[0], .float32), 0, "routed picks");
        try expectClose(got[2], want[1], 0, "routed weights");
        const fed = try kernels.ups(&g, &L, &cfg, x, .{ .picks = try g.reshape(want[0], &.{rows * 8}) });
        try expectClose(fed[0], got[0], 0, "rows from given picks");
    };
}

// Bar: the packed kernels take an affine bank at every width a pack may be re-quantized to, read in place.
// Bar: the pick grouping tiles every pick once, each tile holding at most BM (4 for <= 128 picks, 8 past that) picks of one expert, the used tiles first.
test "inkling2 kernels: the pick grouping tiles each pick once, an expert at a time" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6d1);
    const rng = prng.random();
    const experts = 128;
    for ([_]usize{ 16, 40, 64, 128, 160, 256 }) |n| for ([_]u32{ 1, 3, 9, experts }) |pool| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        var picks: [256]i32 = undefined;
        for (picks[0..n]) |*p| p.* = @intCast(rng.uintLessThan(u32, pool));
        const sel = try g.astype(try g.fromSlice(i32, picks[0..n], &.{@intCast(n)}), .uint32);
        const bm: usize = @intCast(kernels.groupTileWidth(@intCast(n)));
        const grp = try kernels.groupPicks(&g, sel, experts);
        try ops.evalMany(&grp);
        const max_tiles: usize = @intCast(ops.shape(grp[0])[0]);
        const tile_expert = mlx.mlx_array_data_uint32(grp[0]).?[0..max_tiles];
        const tile_picks = mlx.mlx_array_data_uint32(grp[1]).?[0..max_tiles];
        const pick_at = mlx.mlx_array_data_uint32(grp[2]).?[0 .. bm * max_tiles];
        var seen: [256]bool = @splat(false);
        var tiles: usize = 0;
        for (0..max_tiles) |t| {
            if (tile_expert[t] == 0xFFFFFFFF) {
                try testing.expectEqual(@as(u32, 0), tile_picks[t]);
                continue;
            }
            // The used tiles come first.
            try testing.expectEqual(t, tiles);
            tiles += 1;
            try testing.expect(tile_picks[t] >= 1 and tile_picks[t] <= bm);
            for (pick_at[bm * t ..][0..tile_picks[t]]) |p| {
                try testing.expect(p < n and !seen[p]);
                seen[p] = true;
                try testing.expectEqual(tile_expert[t], @as(u32, @intCast(picks[p])));
            }
        }
        for (seen[0..n]) |x| try testing.expect(x);
        var count: [experts]usize = @splat(0);
        for (picks[0..n]) |p| count[@intCast(p)] += 1;
        var want: usize = 0;
        for (count) |c| want += (c + bm - 1) / bm;
        try testing.expectEqual(want, tiles);
    };
}

test "inkling2 kernels: an affine bank is a packed format at every width" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var seed: u64 = 14000;
    for (affine_bits) |bits| for (affine_groups) |group| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const bank = try packedBank(&g, a, seed, &.{ 4, 64, 256 }, bits, group, .affine);
        try testing.expect(kernels.packedFormat(bank, 256) != null);
    };
}

// Bar: an affine expert bank at every width a pack may be re-quantized to (2-6 and 8 bits, groups 32, 64 and 128)
// gathers as the dequantized bank does, unsorted (a row's picks) and sorted (a prompt's), f32 rows: 1e-3 of unit RMS.
test "inkling2 experts: an affine bank gathers at every width as its dequantized rows do" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var seed: u64 = 12000;
    for (affine_bits) |bits| for (affine_groups) |group| for ([_]bool{ false, true }) |sorted| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const in: c_int = @intCast(4 * group);
        const bank = try packedBank(&g, a, seed, &.{ 8, 64, in }, bits, group, .affine);
        const x = try randomRow(&g, a, seed + 1, &.{ 6, 1, in }, 1.0, .float32);
        const picks = try g.fromSlice(u32, &.{ 5, 5, 0, 3, 3, 7 }, &.{6});
        const got = try layers.gather(&g, x, &bank, picks, sorted);
        var dq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_dequantize(&dq, bank.w, bank.s, bank.b, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, g.s));
        const dense = try g.adopt(dq);
        const want = try g.matmul(x, try g.transpose(try g.take(dense, picks, 0), &.{ 0, 2, 1 }));
        var label: [64]u8 = undefined;
        try expectClose(got, want, 1e-3, try std.fmt.bufPrint(&label, "affine {d} bits, group {d}, sorted {}", .{ bits, group, sorted }));
    };
}

// Bar: only an affine bank on a GPU before Apple9 (M1 g13, M2 g14), which has no bf16 ALU, reads bf16 rows as fp16.
test "inkling2 experts: bf16 rows go into the gather as fp16 for an affine bank on a GPU without bf16 only" {
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.bfloat16, .affine, true, 13));
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.bfloat16, .affine, true, 14));
    for ([_]u32{ 15, 16, 17, 18 }) |gen| try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, true, gen));
    // The release's banks (raw outputs past fp16), an unknown generation, a CPU stream and every other dtype stay.
    for ([_]QuantMode{ .nvfp4, .mxfp4, .mxfp8 }) |mode| try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, mode, true, 13));
    try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, true, 0));
    try testing.expectEqual(ops.Dtype.bfloat16, layers.gatherRowsDtype(.bfloat16, .affine, false, 13));
    try testing.expectEqual(ops.Dtype.float32, layers.gatherRowsDtype(.float32, .affine, true, 13));
    try testing.expectEqual(ops.Dtype.float16, layers.gatherRowsDtype(.float16, .affine, true, 13));
}

// Bar: a bf16 gather served as fp16 (device generation pinned to an M1's) leaves bf16 and agrees with the plain bf16
// gather to a bf16 step, (an NVFP4 bank, which keeps bf16 rows, as a control), sorted and not.
test "inkling2 experts: the fp16 gather of a bf16 call agrees with the bf16 one" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const saved = device.device_gen_override;
    defer device.device_gen_override = saved;
    var seed: u64 = 13000;
    for ([_]ExpertFormat{ .{ .mode = .affine, .bits = 4, .group = 64 }, .{ .mode = .affine, .bits = 3, .group = 64 }, .{ .mode = .nvfp4, .bits = 4, .group = 16 } }) |f| for ([_]bool{ false, true }) |sorted| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const bank = try packedBank(&g, a, seed, &.{ 8, 64, 256 }, f.bits, f.group, f.mode.?);
        const x = try randomRow(&g, a, seed + 1, &.{ 6, 1, 256 }, 1.0, .bfloat16);
        const picks = try g.fromSlice(u32, &.{ 5, 5, 0, 3, 3, 7 }, &.{6});
        device.device_gen_override = .{ .gen = 17, .phone = false };
        const plain = try layers.gather(&g, x, &bank, picks, sorted);
        device.device_gen_override = .{ .gen = 13, .phone = false };
        const served = try layers.gather(&g, x, &bank, picks, sorted);
        try testing.expectEqual(ops.Dtype.bfloat16, ops.dtype(served));
        var label: [64]u8 = undefined;
        try expectClose(try g.astype(served, .float32), try g.astype(plain, .float32), 0.1, try std.fmt.bufPrint(&label, "{s} {d} bits, sorted {}", .{ @tagName(f.mode.?), f.bits, sorted }));
    };
}

// Bar: on an M1 the fp16 gather of a bf16 call is MLX's fp16 gather bit for bit: bf16 scales beside fp16 rows would
// promote it to f32, the kernel the fp16 rows are there to avoid. Sorted and not, two widths.
test "inkling2 experts: the fp16 gather of a bf16 call reads its scales and biases as fp16" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const saved = device.device_gen_override;
    defer device.device_gen_override = saved;
    device.device_gen_override = .{ .gen = 13, .phone = false };
    var seed: u64 = 14000;
    for ([_]u32{ 3, 4 }) |bits| for ([_]bool{ false, true }) |sorted| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        // bf16 scales and biases, as a pack's loader leaves them (`narrowsLoadedF16`).
        var bank = try packedBank(&g, a, seed, &.{ 8, 64, 256 }, bits, 64, .affine);
        bank.s = try g.astype(bank.s, .bfloat16);
        bank.b = try g.astype(bank.b, .bfloat16);
        const x = try randomRow(&g, a, seed + 1, &.{ 6, 1, 256 }, 1.0, .bfloat16);
        const picks = try g.fromSlice(u32, &.{ 0, 3, 3, 5, 5, 7 }, &.{6});
        const served = try layers.gather(&g, x, &bank, picks, sorted);
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&y, try g.astype(x, .float16), bank.w, try g.astype(bank.s, .float16), try g.astype(bank.b, .float16), .{ .ctx = null }, picks, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", sorted, g.s));
        const want = try g.astype(try g.adopt(y), .bfloat16);
        var label: [48]u8 = undefined;
        try expectClose(try g.astype(served, .float32), try g.astype(want, .float32), 0.0, try std.fmt.bufPrint(&label, "affine {d} bits, sorted {}", .{ bits, sorted }));
    };
}

// Bar: the flash kernel (q, k, v as halves, an online softmax from the sink) against the masked SDPA chain on the
// same rows, on a sliding and a global layer, from the sequence's start and behind earlier keys (past the window, the
// extent and the log-scaling floor), for a row count off the kernel's tiles: f32 1e-2, bf16 5e-2 of unit-RMS values.
test "inkling2 kernels: the relative attention kernel reads a prompt's rows as the masked chain does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 12000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ 5, 43 }) |rows| for ([_]c_int{ 0, 3, 50 }) |offset| for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const L = &s.body.layers[li];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        // The chain's rows are a transposed view: compared as the dense array `attend` makes of them.
        // Both arms: the flash kernel, and under `small_rows` the mask kernel with MLX's attention.
        const want = try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset));
        for ([_]bool{ true, false }) |flash| {
            try expectClose(try g.contiguous(try kernels.relative(&g, L, cfg, p.q, p.rel, p.sinks, k, v, offset, flash)), want, if (dt == .float32) 1e-2 else 5e-2, "relative read");
        }
    };
}

// Bar: a prompt's rows read on the tensor unit against the masked chain, on a sliding and a global layer, at row counts
// that leave a partial 16-row tile and a partial 64-row group, from the start and behind earlier keys (past the window,
// the 1024-key bias extent, the 32-key tile and the log-scaling floor): 5e-2 of unit-RMS values in half dtypes.
test "inkling2 kernels: a prompt's rows read on the tensor unit are the masked chain's" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 13000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ 33, 70, 200 }) |rows| for ([_]c_int{ 0, 5, 1100, 2500 }) |offset| for ([_]ops.Dtype{ .float16, .bfloat16 }) |dt| {
        const L = &s.body.layers[li];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        const want = try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset));
        const got = (try kernels.relativeNax(&g, L, cfg, p.q, p.rel, p.sinks, k, v, offset)) orelse return error.TestUnexpectedResult;
        try expectClose(try g.contiguous(got), want, 5e-2, "tensor-unit read");
    };
}

// Bar: the split read (keys split over threadgroups, combined under the gate) against the masked chain times its
// gate, on a sliding and a global layer, from the start and behind earlier keys (past the window, the extent and the
// log-scaling floor), at the fewest rows it takes, an odd count and the most: 5e-2 of unit-RMS values in half dtypes.
test "inkling2 kernels: the split read is the masked chain's, gated" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 13000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ kernels.split_from, 5, kernels.split_rows }) |rows| for ([_]c_int{ 0, 3, 300 }) |offset| for ([_]ops.Dtype{ .bfloat16, .float16 }) |dt| {
        const L = &s.body.layers[li];
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 5, &.{rows * heads * hd}, 2.0, .float32));
        const want = try kernels.gated(&g, try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset)), gate, rows);
        const got = (try kernels.splitRead(&g, L, cfg, p.q, p.rel, p.sinks, k, v, null, offset, gate)) orelse return error.TestUnexpectedResult;
        try expectClose(got, want, 5e-2, "split read");
    };
}

// Bar: a verify's rows read the cache in place with their own keys behind it (two sources, no copy) as the masked
// chain reads the joined keys: from 2 rows to 16 (two fragments), on a sliding and a global layer, to 5e-2.
test "inkling2 kernels: a verify's split read of the cache and its own rows is the masked chain's over both" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Stage.init(a);
    defer s.deinit();
    const cfg = &s.body.cfg;
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const kv_heads: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    var seed: u64 = 23000;
    for ([_]usize{ 0, 1 }) |li| for ([_]c_int{ 2, 3, kernels.split_rows, 9, 13, kernels.open_rows }) |rows| for ([_]c_int{ 1, 300 }) |offset| {
        const L = &s.body.layers[li];
        const dt: ops.Dtype = .bfloat16;
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 10;
        const total = offset + rows;
        const keys = if (L.global) total else @min(total, @as(c_int, @intCast(cfg.sliding_window)) + rows - 1);
        const q = try randomRow(&g, a, seed, &.{ 1, rows, heads, hd }, 0.6, dt);
        const p: layers.Pre = .{
            .q = q,
            .k = q,
            .v = q,
            .rel = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, @intCast(cfg.d_rel) }, 0.5, dt),
            .gate = q,
            .sinks = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads }, 1.0, dt),
            .k_state = q,
            .v_state = q,
            .stream = q,
            .normed = q,
        };
        const k = try randomRow(&g, a, seed + 3, &.{ 1, kv_heads, keys, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, keys, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 5, &.{rows * heads * hd}, 2.0, .float32));
        const want = try kernels.gated(&g, try g.contiguous(try layers.wideRead(&g, L, cfg, &p, k, v, offset)), gate, rows);
        const own = [2]ops.Array{ try g.sliceAxis(k, 2, keys - rows, keys), try g.sliceAxis(v, 2, keys - rows, keys) };
        const got = (try kernels.splitRead(&g, L, cfg, p.q, p.rel, p.sinks, try g.sliceAxis(k, 2, 0, keys - rows), try g.sliceAxis(v, 2, 0, keys - rows), own, offset, gate)) orelse return error.TestUnexpectedResult;
        try expectClose(got, want, 5e-2, "split read in place");
    };
}

// Bar: `relsplit_nax.metal` (single-pass 2..16-row NAX split-K read) matches `splitRead` on real geometry (hd=128,
// 24Q/8KV) across sliding and global layers, direct (1 split) and multi-split, up to 16 rows.
test "inkling2 kernels: relsplit_nax matches splitRead" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const heads: c_int = 24;
    const kv_heads: c_int = 8;
    const hd: c_int = 128;
    const d_rel: c_int = 16;
    const r_ext: c_int = 1024;
    var cfg: plugin.config.Config = undefined;
    cfg.hidden_size = 2304;
    cfg.num_attention_heads = @intCast(heads);
    cfg.num_key_value_heads = @intCast(kv_heads);
    cfg.head_dim = @intCast(hd);
    cfg.d_rel = @intCast(d_rel);
    cfg.sliding_window = 1024;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.2;
    var seed: u64 = 51000;
    for ([_]bool{ false, true }) |global| for ([_]c_int{ 2, 5, 8, 12, 15, 16 }) |rows| for ([_]c_int{ 4, 50, 600, 1200 }) |offset| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const dt: ops.Dtype = .bfloat16;
        const rel_proj = try randomRow(&g, a, seed, &.{ d_rel, r_ext }, 0.3, dt);
        var L: layers.Layer = undefined;
        L.global = global;
        L.rel_proj = rel_proj;
        const cached = if (global) offset else @min(offset, 1023);
        const q = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, hd }, 0.6, dt);
        const rel_st = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads, d_rel }, 0.5, dt);
        const sinks = try randomRow(&g, a, seed + 3, &.{ 1, rows, heads }, 1.0, dt);
        const k = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, cached, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 5, &.{ 1, kv_heads, cached, hd }, 1.0, dt);
        const kb = try randomRow(&g, a, seed + 6, &.{ 1, kv_heads, rows, hd }, 0.6, dt);
        const vb = try randomRow(&g, a, seed + 7, &.{ 1, kv_heads, rows, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 8, &.{rows * heads * hd}, 2.0, .float32));

        const want = (try kernels.splitRead(&g, &L, &cfg, q, rel_st, sinks, k, v, .{ kb, vb }, offset, gate)) orelse return error.TestUnexpectedResult;
        const got = (try kernels.splitReadNax(&g, &L, &cfg, q, rel_st, sinks, k, v, .{ kb, vb }, offset, gate)) orelse return error.TestUnexpectedResult;
        try expectClose(got, want, 5e-2, "relsplit_nax chain");
    };
}

// Bar: `relsplit_nax` with `TREE = 1` matches `relsplit_lanes` with `TREE = 1` on real geometry (hd=128, 24Q/8KV)
// across sliding and global layers, direct (1 split) and multi-split, 2 and 16 rows on a branching tree.
test "inkling2 kernels: relsplit_nax TREE=1 matches relsplit_lanes TREE=1" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const heads: c_int = 24;
    const kv_heads: c_int = 8;
    const hd: c_int = 128;
    const d_rel: c_int = 16;
    const r_ext: c_int = 1024;
    var cfg: plugin.config.Config = undefined;
    cfg.hidden_size = 2304;
    cfg.num_attention_heads = @intCast(heads);
    cfg.num_key_value_heads = @intCast(kv_heads);
    cfg.head_dim = @intCast(hd);
    cfg.d_rel = @intCast(d_rel);
    cfg.sliding_window = 1024;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.2;
    // Branching 16-node tree parent array (0 root; branches at 0, 1, 2, 4, 5).
    const par16 = [16]i32{ -1, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 10, 12 };
    var meta16: [32]i32 = undefined;
    for (0..16) |r| {
        if (r == 0) {
            meta16[0] = 0;
            meta16[1] = 1;
        } else {
            const p: usize = @intCast(par16[r]);
            meta16[r * 2 + 0] = meta16[p * 2 + 0] + 1;
            meta16[r * 2 + 1] = meta16[p * 2 + 1] | (@as(i32, 1) << @intCast(r));
        }
    }
    var seed: u64 = 91000;
    for ([_]bool{ false, true }) |global| for ([_]c_int{ 2, 16 }) |rows| for ([_]c_int{ 4, 1200 }) |offset| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const dt: ops.Dtype = .bfloat16;
        var L: layers.Layer = undefined;
        L.global = global;
        L.rel_proj = try randomRow(&g, a, seed, &.{ d_rel, r_ext }, 0.3, dt);
        const cached = if (global) offset else @min(offset, 1023);
        const q = try randomRow(&g, a, seed + 1, &.{ 1, rows, heads, hd }, 0.6, dt);
        const rel_st = try randomRow(&g, a, seed + 2, &.{ 1, rows, heads, d_rel }, 0.5, dt);
        const sinks = try randomRow(&g, a, seed + 3, &.{ 1, rows, heads }, 1.0, dt);
        const k = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, cached, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 5, &.{ 1, kv_heads, cached, hd }, 1.0, dt);
        const kb = try randomRow(&g, a, seed + 6, &.{ 1, kv_heads, rows, hd }, 0.6, dt);
        const vb = try randomRow(&g, a, seed + 7, &.{ 1, kv_heads, rows, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 8, &.{rows * heads * hd}, 2.0, .float32));
        const tmeta = try g.fromSlice(i32, meta16[0 .. @as(usize, @intCast(rows)) * 2], &.{ rows, 2 });
        const want = (try kernels.splitReadLanesM4(&g, &L, &cfg, q, rel_st, sinks, &.{k}, &.{v}, .{ kb, vb }, &.{offset}, gate, rows, tmeta)) orelse return error.TestUnexpectedResult;
        const got = (try kernels.splitReadNaxLanesTree(&g, &L, &cfg, q, rel_st, sinks, &.{k}, &.{v}, .{ kb, vb }, &.{offset}, gate, rows, tmeta)) orelse return error.TestUnexpectedResult;
        try expectClose(got, want, 6e-2, "relsplit_nax TREE=1");
    };
}

// Bar: the NAX read of several slots' blocks in launches four slots wide is each slot's block read alone (`splitRead` over the
// window the slot's rows reach), sliding and global, with slots whose caches differ in length and span one split or several.
test "inkling2 kernels: relsplit_nax reads several slots' blocks as each slot alone" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const heads: c_int = 24;
    const kv_heads: c_int = 8;
    const hd: c_int = 128;
    const d_rel: c_int = 16;
    const r_ext: c_int = 1024;
    var cfg: plugin.config.Config = undefined;
    cfg.hidden_size = 2304;
    cfg.num_attention_heads = @intCast(heads);
    cfg.num_key_value_heads = @intCast(kv_heads);
    cfg.head_dim = @intCast(hd);
    cfg.d_rel = @intCast(d_rel);
    cfg.sliding_window = 1024;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.2;
    const cached_at = [_]c_int{ 40, 700, 1500, 130, 2100, 900 };
    var seed: u64 = 73000;
    for ([_]bool{ false, true }) |global| for ([_]c_int{ 3, 8 }) |rows| for ([_]usize{ 3, 6 }) |n| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const dt: ops.Dtype = .bfloat16;
        var L: layers.Layer = undefined;
        L.global = global;
        L.rel_proj = try randomRow(&g, a, seed, &.{ d_rel, r_ext }, 0.3, dt);
        const total: c_int = rows * @as(c_int, @intCast(n));
        const q = try randomRow(&g, a, seed + 1, &.{ 1, total, heads, hd }, 0.6, dt);
        const rel_st = try randomRow(&g, a, seed + 2, &.{ 1, total, heads, d_rel }, 0.5, dt);
        const sinks = try randomRow(&g, a, seed + 3, &.{ 1, total, heads }, 1.0, dt);
        const kb = try randomRow(&g, a, seed + 6, &.{ 1, kv_heads, total, hd }, 0.6, dt);
        const vb = try randomRow(&g, a, seed + 7, &.{ 1, kv_heads, total, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 8, &.{total * heads * hd}, 2.0, .float32));
        var ks: [6]ops.Array = undefined;
        var vs: [6]ops.Array = undefined;
        var offsets: [6]c_int = undefined;
        var want: [6]ops.Array = undefined;
        for (0..n) |i| {
            const cached = cached_at[i];
            ks[i] = try randomRow(&g, a, seed + 30 + i, &.{ 1, kv_heads, cached, hd }, 0.6, dt);
            vs[i] = try randomRow(&g, a, seed + 40 + i, &.{ 1, kv_heads, cached, hd }, 1.0, dt);
            offsets[i] = cached;
            const at: c_int = @as(c_int, @intCast(i)) * rows;
            // A sliding slot's rows reach at most `window - 1` keys before the block.
            const from: c_int = if (global) 0 else @max(0, cached - 1023);
            want[i] = (try kernels.splitRead(&g, &L, &cfg, try g.sliceAxis(q, 1, at, at + rows), try g.sliceAxis(rel_st, 1, at, at + rows), try g.sliceAxis(sinks, 1, at, at + rows), try g.sliceAxis(ks[i], 2, from, cached), try g.sliceAxis(vs[i], 2, from, cached), .{ try g.sliceAxis(kb, 2, at, at + rows), try g.sliceAxis(vb, 2, at, at + rows) }, cached, try g.sliceAxis(gate, 0, at * heads * hd, (at + rows) * heads * hd))) orelse return error.TestUnexpectedResult;
        }
        const got = (try kernels.splitReadNaxLanes(&g, &L, &cfg, q, rel_st, sinks, ks[0..n], vs[0..n], .{ kb, vb }, offsets[0..n], gate, rows)) orelse return error.TestUnexpectedResult;
        try expectClose(got, try g.concat(want[0..n], 1), 5e-2, "relsplit_nax lanes");
    };
}

// Bar: the drafter's Markov chain as three launches a step drafts the op chain's tokens, each with its probability to
// 1e-2 (one bf16 ulp of the top logit moves it a few percent), from three committed tokens over random heads whose
// logits stand apart (a pick a bf16 rounding cannot flip); in both, the probability reads the soft-capped logits.
test "inkling2 kernels: the fused Markov chain drafts the op chain's tokens" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const rank = 64;
    const hidden = 128;
    const vpad = 4096;
    const vocab = 4000;
    const steps = 7;
    for ([_]ops.Dtype{ .bfloat16, .float16 }) |dt| for ([_]i32{ 3, 1777, 3999 }) |first| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const seed: u64 = 15000 + @as(u64, @intCast(first));
        var d: draft.Drafter = .{
            .cfg = .{ .layers = 0, .heads = 1, .kv_heads = 1, .head_dim = 64, .intermediate = 1, .window = 2, .block = steps + 1, .rank = rank, .tap_count = 1 },
            .hidden = hidden,
            .s = fx_stream,
        };
        d.norm = try g.ones(&.{hidden}, dt);
        d.prev_rows = try randomRow(&g, a, seed, &.{ vpad, rank }, 0.5, dt);
        d.rnn = .{ .w = try randomRow(&g, a, seed + 1, &.{ 3 * rank, 2 * rank + 2 * hidden }, 0.05, dt) };
        d.rnn_bias = try randomRow(&g, a, seed + 2, &.{3 * rank}, 0.2, dt);
        d.fold = try randomRow(&g, a, seed + 3, &.{ rank, vpad }, 0.3, dt);
        d.fold_bias = try randomRow(&g, a, seed + 4, &.{ 1, 1, vpad }, 0.3, dt);
        const rows = try randomRow(&g, a, seed + 5, &.{ 1, steps, hidden }, 1.0, dt);
        const newest = try randomRow(&g, a, seed + 6, &.{ 1, 1, hidden }, 1.0, dt);
        const base = try g.astype(try randomRow(&g, a, seed + 7, &.{ 1, steps, vpad }, 4.0, dt), .float32);
        const t1 = try g.fromSlice(i32, &.{first}, &.{1});
        try testing.expect(kernels.markovRuns(&g, d.rnn.w, d.fold, rank));
        const want = try d.markov(&g, t1, rows, newest, base, vocab, 30, false, null);
        const got = try d.markov(&g, t1, rows, newest, base, vocab, 30, true, null);
        try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
        try expectClose(got.confidence, want.confidence, 1e-2, "draft probabilities");
        // Logits far apart: their own top is all but sure, their capped top is not (the head caps at 30).
        const far = try g.mulScalar(base, 25);
        for ([_]bool{ false, true }) |fused| {
            const raw = try hostValues(a, &g, (try d.markov(&g, t1, rows, newest, far, vocab, 0, fused, null)).confidence);
            defer a.free(raw);
            const cap = try hostValues(a, &g, (try d.markov(&g, t1, rows, newest, far, vocab, 30, fused, null)).confidence);
            defer a.free(cap);
            try testing.expect(raw[0] > 0.5 and cap[0] < 0.1);
        }
    };
}

// Bar: drafts drawn under a law come from the drafter's soft-capped softmax at that temperature (the CPU stream, f32):
// each law row sums to one, its token has mass, a draft's confidence is its row's peak, and near zero temperature the
// draws are the argmax drafts.
// Bar: the drafter's 8-bit copy of a dense head reads f32 rows as the head does, to 2.5e-2 where the logits spread
// 0.6 (256 levels over each group of 64 weights); a head that ships quantized is read as it is (no copy).
test "inkling2 draft: the drafter's 8-bit head reads rows as the dense head does" {
    const a = testing.allocator;
    fx_stream = mlx.mlx_default_cpu_stream_new();
    var g = ops.Graph.init(a, fx_stream);
    defer g.deinit();
    const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
    const dense: Linear = .{ .w = try randomRow(&g, a, 4100, &.{ 512, 128 }, 0.05, .bfloat16) };
    const copy = (try draft.headCopy(&g, &dense)).?;
    const rows = try randomRow(&g, a, 4101, &.{ 1, 5, 128 }, 1.0, .float32);
    try expectClose(try layers.projectRows(&g, &copy, rows), try layers.projectRows(&g, &dense, rows), 2.5e-2, "8-bit head rows");
    try testing.expect((try draft.headCopy(&g, &copy)) == null);
}

test "inkling2 draft: a sampled chain draws from the law it returns" {
    const a = testing.allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const saved = fx_stream;
    fx_stream = cpu;
    defer fx_stream = saved;
    var g = ops.Graph.init(a, cpu);
    defer g.deinit();
    const rank = 8;
    const hidden = 32;
    const vocab = 64;
    const steps = 5;
    const dt: ops.Dtype = .float32;
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 0, .heads = 1, .kv_heads = 1, .head_dim = 16, .intermediate = 1, .window = 2, .block = steps + 1, .rank = rank, .tap_count = 1 },
        .hidden = hidden,
        .s = cpu,
    };
    d.norm = try g.ones(&.{hidden}, dt);
    d.prev_rows = try randomRow(&g, a, 19001, &.{ vocab, rank }, 0.5, dt);
    d.rnn = .{ .w = try randomRow(&g, a, 19002, &.{ 3 * rank, 2 * rank + 2 * hidden }, 0.1, dt) };
    d.rnn_bias = try randomRow(&g, a, 19003, &.{3 * rank}, 0.2, dt);
    d.fold = try randomRow(&g, a, 19004, &.{ rank, vocab }, 0.3, dt);
    d.fold_bias = try randomRow(&g, a, 19005, &.{ 1, 1, vocab }, 0.3, dt);
    const rows = try randomRow(&g, a, 19006, &.{ 1, steps, hidden }, 1.0, dt);
    const newest = try randomRow(&g, a, 19007, &.{ 1, 1, hidden }, 1.0, dt);
    const base = try randomRow(&g, a, 19008, &.{ 1, steps, vocab }, 2.0, dt);
    const t1 = try g.fromSlice(i32, &.{3}, &.{1});
    const argmax = try d.markov(&g, t1, rows, newest, base, vocab, 30, false, null);
    const cold = try d.markov(&g, t1, rows, newest, base, vocab, 30, false, .{ .temperature = 1e-4, .seed = 7 });
    try expectClose(cold.tokens, argmax.tokens, 0, "near-greedy draws");
    const warm = try d.markov(&g, t1, rows, newest, base, vocab, 30, false, .{ .temperature = 1.0, .seed = 7 });
    const law = try hostValues(a, &g, warm.probs);
    defer a.free(law);
    const sure = try hostValues(a, &g, warm.confidence);
    defer a.free(sure);
    const ids = try hostValues(a, &g, try g.astype(warm.tokens, .float32));
    defer a.free(ids);
    try testing.expectEqual(@as(usize, steps * vocab), law.len);
    for (0..steps) |i| {
        const row = law[i * vocab ..][0..vocab];
        var sum: f32 = 0;
        for (row) |x| sum += x;
        try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-5);
        try testing.expect(row[@intFromFloat(ids[i])] > 0);
        try testing.expectEqual(std.mem.max(f32, row), sure[i]);
    }
}

// Bar: a slot's rounds with the drafter's context keys and values carried over draft what rounds projecting the
// context afresh draft, the probabilities to 1e-5 (the CPU stream, f32): a ring that grows, slides past its keep, holds
// still (no new row) and jumps past the cache (projected afresh).
test "inkling2 draft: rounds that carry the context keys and values draft what fresh ones do" {
    const a = testing.allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const saved = fx_stream;
    fx_stream = cpu;
    defer fx_stream = saved;
    var g = ops.Graph.init(a, cpu);
    defer g.deinit();
    const H = 32;
    const HD = 16;
    const V = 64;
    const R = 8;
    const dt: ops.Dtype = .float32;
    var seed: u64 = 17000;
    const W = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, shape: []const c_int, scale: f32) !ops.Array {
            sd.* += 1;
            return randomRow(gr, al, sd.*, shape, scale, .float32);
        }
    };
    const lin = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, out: c_int, in: c_int) !@import("../../sdk/bases/blocks/linear.zig").Linear {
            return .{ .w = try W.of(gr, al, sd, &.{ out, in }, 0.2) };
        }
    };
    const one = try g.ones(&.{H}, dt);
    var layer: draft.Layer = .{
        .in_norm = one,
        .post_norm = one,
        .mix_norm = one,
        .q = try lin.of(&g, a, &seed, 2 * HD, H),
        .k = try lin.of(&g, a, &seed, HD, H),
        .v = try lin.of(&g, a, &seed, HD, H),
        .o = try lin.of(&g, a, &seed, H, 2 * HD),
        .k_ctx = try lin.of(&g, a, &seed, HD, H),
        .v_ctx = try lin.of(&g, a, &seed, HD, H),
        .q_norm = try g.ones(&.{HD}, dt),
        .k_norm = try g.ones(&.{HD}, dt),
        .k_norm_ctx = try g.ones(&.{HD}, dt),
        .gate = try lin.of(&g, a, &seed, 48, H),
        .up = try lin.of(&g, a, &seed, 48, H),
        .down = try lin.of(&g, a, &seed, H, 48),
    };
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 1, .heads = 2, .kv_heads = 1, .head_dim = HD, .intermediate = 48, .window = 13, .block = 4, .rank = R, .tap_count = 2 },
        .hidden = H,
        .s = cpu,
    };
    d.layers = (&layer)[0..1];
    d.norm = one;
    d.mask = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    d.alpha = try g.softmax(try W.of(&g, a, &seed, &.{ 2, 2, 1 }, 1.0), 1, true);
    d.prev_rows = try W.of(&g, a, &seed, &.{ V, R }, 0.5);
    d.rnn = .{ .w = try W.of(&g, a, &seed, &.{ 3 * R, 2 * R + 2 * H }, 0.1) };
    d.rnn_bias = try W.of(&g, a, &seed, &.{3 * R}, 0.1);
    d.rnn_norm = one;
    d.fold = try W.of(&g, a, &seed, &.{ R, V }, 0.3);
    d.fold_bias = try W.of(&g, a, &seed, &.{ 1, 1, V }, 0.3);
    const head = try lin.of(&g, a, &seed, V, H);
    const anchor = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    const t1 = try g.fromSlice(i32, &.{7}, &.{1});
    const full = try W.of(&g, a, &seed, &.{ 1, 40, 2 * H }, 1.0);
    var cache: draft.ContextKv = .{};
    defer cache.release();
    // The ring's rows [lo, hi), ending at the round's position: grows, slides past 12 rows, holds, jumps.
    for ([_][2]c_int{ .{ 0, 5 }, .{ 0, 9 }, .{ 2, 14 }, .{ 6, 18 }, .{ 6, 18 }, .{ 28, 40 } }) |span| {
        const ring = try g.sliceAxis(full, 1, span[0], span[1]);
        const got = try d.draft(&g, t1, anchor, ring, span[1], &head, V - 4, 30, &cache, null);
        const want = try d.draft(&g, t1, anchor, ring, span[1], &head, V - 4, 30, null, null);
        try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
        try expectClose(got.confidence, want.confidence, 1e-5, "draft probabilities");
    }
}

// Bar: a full layer's context appended a forward at a time, its buffers grown in place, drafts what one projection of
// every row does, past the ring's reach; beside a sliding layer, a carried context drafts what a fresh one does; the
// tokens exact, the probabilities to 1e-5.
test "inkling2 draft: a layer outside local_layer_ids reads every committed position" {
    const a = testing.allocator;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const saved = fx_stream;
    fx_stream = cpu;
    defer fx_stream = saved;
    var g = ops.Graph.init(a, cpu);
    defer g.deinit();
    const H = 32;
    const HD = 16;
    const V = 64;
    const R = 8;
    const dt: ops.Dtype = .float32;
    var seed: u64 = 18000;
    const W = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, shape: []const c_int, scale: f32) !ops.Array {
            sd.* += 1;
            return randomRow(gr, al, sd.*, shape, scale, .float32);
        }
        fn lin(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, out: c_int, in: c_int) !@import("../../sdk/bases/blocks/linear.zig").Linear {
            return .{ .w = try of(gr, al, sd, &.{ out, in }, 0.2) };
        }
        fn layer(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, one: ops.Array, ones_hd: ops.Array) !draft.Layer {
            return .{
                .in_norm = one,
                .post_norm = one,
                .mix_norm = one,
                .q = try lin(gr, al, sd, 2 * HD, H),
                .k = try lin(gr, al, sd, HD, H),
                .v = try lin(gr, al, sd, HD, H),
                .o = try lin(gr, al, sd, H, 2 * HD),
                .k_ctx = try lin(gr, al, sd, HD, H),
                .v_ctx = try lin(gr, al, sd, HD, H),
                .q_norm = ones_hd,
                .k_norm = ones_hd,
                .k_norm_ctx = ones_hd,
                .gate = try lin(gr, al, sd, 48, H),
                .up = try lin(gr, al, sd, 48, H),
                .down = try lin(gr, al, sd, H, 48),
            };
        }
    };
    const one = try g.ones(&.{H}, dt);
    const ones_hd = try g.ones(&.{HD}, dt);
    var two = [_]draft.Layer{ try W.layer(&g, a, &seed, one, ones_hd), try W.layer(&g, a, &seed, one, ones_hd) };
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 2, .heads = 2, .kv_heads = 1, .head_dim = HD, .intermediate = 48, .window = 13, .block = 4, .rank = R, .tap_count = 2, .local = 0 },
        .hidden = H,
        .s = cpu,
    };
    d.layers = two[0..];
    d.norm = one;
    d.mask = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    d.alpha = try g.softmax(try W.of(&g, a, &seed, &.{ 3, 2, 1 }, 1.0), 1, true);
    d.prev_rows = try W.of(&g, a, &seed, &.{ V, R }, 0.5);
    d.rnn = .{ .w = try W.of(&g, a, &seed, &.{ 3 * R, 2 * R + 2 * H }, 0.1) };
    d.rnn_bias = try W.of(&g, a, &seed, &.{3 * R}, 0.1);
    d.rnn_norm = one;
    d.fold = try W.of(&g, a, &seed, &.{ R, V }, 0.3);
    d.fold_bias = try W.of(&g, a, &seed, &.{ 1, 1, V }, 0.3);
    const head = try W.lin(&g, a, &seed, V, H);
    const anchor = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
    const t1 = try g.fromSlice(i32, &.{7}, &.{1});
    const all = try W.of(&g, a, &seed, &.{ 1, 40, 2 * H }, 1.0);
    const keep: c_int = 12;
    // Each round: the rows a forward pushes [from, to), then a draft at `to` from the ring's newest `keep`.
    const pushes = [_][2]c_int{ .{ 0, 5 }, .{ 5, 9 }, .{ 9, 14 }, .{ 14, 18 }, .{ 18, 18 }, .{ 18, 40 } };
    for ([_]?u32{ 0, 1 }) |local| {
        d.cfg.local = local;
        // Buffers grown 8 positions at a time: the pushes cross several growths.
        var carried: draft.ContextKv = .{ .chunk = 8 };
        defer carried.release();
        for (pushes) |p| {
            if (p[1] > p[0]) try d.appendFull(&g, &carried, &.{try g.sliceAxis(all, 1, p[0], p[1])}, @intCast(p[0]));
            const ring = try g.sliceAxis(all, 1, @max(0, p[1] - keep), p[1]);
            const got = try d.draft(&g, t1, anchor, ring, p[1], &head, V - 4, 30, &carried, null);
            var want: draft.Drafter.Block = undefined;
            var fresh: draft.ContextKv = .{};
            defer fresh.release();
            if (local.? == 0) {
                // Every layer full: the draft over the whole context as its ring.
                want = try d.draft(&g, t1, anchor, try g.sliceAxis(all, 1, 0, p[1]), p[1], &head, V - 4, 30, null, null);
            } else {
                // The taps as two parts side by side (a forward's per-tap slices), joined by the append.
                const rows = try g.sliceAxis(all, 1, 0, p[1]);
                try d.appendFull(&g, &fresh, &.{ try g.sliceAxis(rows, 2, 0, H), try g.sliceAxis(rows, 2, H, 2 * H) }, 0);
                want = try d.draft(&g, t1, anchor, ring, p[1], &head, V - 4, 30, &fresh, null);
            }
            try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
            try expectClose(got.confidence, want.confidence, 1e-5, "draft probabilities");
        }
        // Past the ring the full layer changes the draft: the ring-only read differs.
        const newest = try g.sliceAxis(all, 1, 40 - keep, 40);
        const full_read = try d.draft(&g, t1, anchor, newest, 40, &head, V - 4, 30, &carried, null);
        const ring_only = try d.draft(&g, t1, anchor, newest, 40, &head, V - 4, 30, null, null);
        try testing.expect(try parity.maxAbsDiffF32(full_read.confidence, ring_only.confidence, fx_stream) > 1e-4);
    }
    // Full buffers that end before the draft (no push since): the full layer reads the ring, as a fresh context does.
    d.cfg.local = 1;
    var behind: draft.ContextKv = .{ .chunk = 8 };
    defer behind.release();
    try d.appendFull(&g, &behind, &.{try g.sliceAxis(all, 1, 0, 18)}, 0);
    _ = try d.draft(&g, t1, anchor, try g.sliceAxis(all, 1, 18 - keep, 18), 18, &head, V - 4, 30, &behind, null);
    const ring = try g.sliceAxis(all, 1, 22 - keep, 22);
    const got = try d.draft(&g, t1, anchor, ring, 22, &head, V - 4, 30, &behind, null);
    const want = try d.draft(&g, t1, anchor, ring, 22, &head, V - 4, 30, null, null);
    try expectClose(got.tokens, want.tokens, 0, "drafted tokens");
    try expectClose(got.confidence, want.confidence, 1e-5, "draft probabilities");
    // A context cut back to position 30 and extended by other rows drafts what one built from those rows does.
    const other = try W.of(&g, a, &seed, &.{ 1, 10, 2 * H }, 1.0);
    const redone = try g.concat(&.{ try g.sliceAxis(all, 1, 0, 30), other }, 1);
    var cut: draft.ContextKv = .{ .chunk = 8 };
    defer cut.release();
    try d.appendFull(&g, &cut, &.{all}, 0);
    _ = try d.draft(&g, t1, anchor, try g.sliceAxis(all, 1, 40 - keep, 40), 40, &head, V - 4, 30, &cut, null);
    cut.cut(30);
    try d.appendFull(&g, &cut, &.{other}, 30);
    var built: draft.ContextKv = .{ .chunk = 8 };
    defer built.release();
    try d.appendFull(&g, &built, &.{redone}, 0);
    const tail = try g.sliceAxis(redone, 1, 40 - keep, 40);
    const after_cut = try d.draft(&g, t1, anchor, tail, 40, &head, V - 4, 30, &cut, null);
    const from_built = try d.draft(&g, t1, anchor, tail, 40, &head, V - 4, 30, &built, null);
    try expectClose(after_cut.tokens, from_built.tokens, 0, "drafted tokens after a cut");
    try expectClose(after_cut.confidence, from_built.confidence, 1e-5, "draft probabilities after a cut");
}

// Bar: the few-row router kernel's logits are each row's one-row matmul (exact f32; the matmul over all the rows
// at once is only good to ~1e-3), to 1e-5 of the largest.
test "inkling2 kernels: the router kernel's rows are the matmul's" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    for ([_]c_int{ 2, 7 }) |rows| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const x = try randomRow(&g, a, 71, &.{ 1, rows, 192 }, 1.0, .float32);
        const router = try randomRow(&g, a, 72, &.{ 24, 192 }, 0.2, .float32);
        var each: [7]ops.Array = undefined;
        for (each[0..@intCast(rows)], 0..) |*r, i| {
            const at: c_int = @intCast(i);
            r.* = try g.reshape(try g.matmul(try g.sliceAxis(x, 1, at, at + 1), try g.transpose(router, &.{ 1, 0 })), &.{ 1, 24 });
        }
        try expectClose(try kernels.routerRows(&g, x, router), try g.concat(each[0..@intCast(rows)], 0), 1e-5, "router rows");
    }
}

// Bar: the gate's product in f32, rounded once: the chain's bytes.
test "inkling2 kernels: the gated kernel is the attention read times its gate, rounded once" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        const x = try randomRow(&g, a, 31, &.{ 1, 4, 1, 16 }, 2.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, 32, &.{64}, 2.0, .float32));
        const want = try g.astype(try g.mul(try g.astype(try g.reshape(x, &.{ 1, 1, 64 }), .float32), gate), dt);
        try expectClose(try kernels.gated(&g, x, gate, 1), want, 0, "gated rows");
    }
}

// Bar: a batched block pass is what each lane's pass alone is (cosine and scale of the block rows, a lane's rows unmoved by
// its neighbours) over rings of different lengths, on a round that projects its whole context and on the next, which projects only the rows committed since; a batched
// draft's law, per lane, is a distribution over its candidates that holds the token it drew.
test "inkling2 dflash2: a batched draft drafts what each lane drafts alone" {
    const env = @import("../../env.zig");
    const pack_dir = std.mem.span(env.get(.INKLING2_DFLASH2_PACK) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rel = try Release.open(a, pack_dir);
    defer rel.close();
    const d = rel.m.drafter2 orelse return error.SkipZigTest;
    const w = &rel.m.base.weights;
    const taps: c_int = @intCast(d.cfg.tap_count);
    const lens = [_]c_int{ 300, 2047, 1200 };
    const starts = [_]c_int{ 5300, 9000, 4000 };
    const firsts = [_]i32{ 11, 4242, 90000 };
    var g = ops.Graph.init(a, rel.m.base.s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(99);
    var rings: [3]ops.Array = undefined;
    var anchors: [3]ops.Array = undefined;
    var t1: [3]ops.Array = undefined;
    for (&rings, &anchors, &t1, lens, firsts) |*ring, *anchor, *first, len, id| {
        const host = try a.alloc(f32, @as(usize, @intCast(len * taps * 2304)));
        for (host) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
        ring.* = try g.astype(try g.fromSlice(f32, host, &.{ 1, len, taps * 2304 }), .bfloat16);
        const row = try a.alloc(f32, 2304);
        for (row) |*x| x.* = prng.random().floatNorm(f32) * 0.3;
        anchor.* = try g.astype(try g.fromSlice(f32, row, &.{ 1, 1, 2304 }), .bfloat16);
        first.* = try g.fromSlice(i32, &.{id}, &.{1});
    }
    var alone: [3]draft.ContextKv = @splat(.{});
    var batch: [3]draft.ContextKv = @splat(.{});
    // A lane's rows do not move with its neighbours: the same lane beside itself and beside another lane's ring and position.
    {
        var rows: [2]ops.Array = undefined;
        for ([_]usize{ 1, 2 }, &rows) |other, *out| {
            var c0: draft.ContextKv = .{};
            var c1: draft.ContextKv = .{};
            defer c0.release();
            defer c1.release();
            const nz0 = try g.concat(&.{ anchors[0], try g.broadcast(d.mask, &.{ 1, 7, 2304 }) }, 1);
            const nz1 = try g.concat(&.{ anchors[other], try g.broadcast(d.mask, &.{ 1, 7, 2304 }) }, 1);
            const got = try d.blockRowsBatch(&g, try g.concat(&.{ nz0, nz1 }, 0), &.{ .{ .taps = rings[0], .position = starts[0], .cache = &c0 }, .{ .taps = rings[other], .position = starts[other], .cache = &c1 } });
            const y = try g.astype(try g.sliceAxis(got, 0, 0, 1), .float32);
            try ops.eval(y);
            out.* = y;
        }
        const x = mlx.mlx_array_data_float32(rows[0]).?[0 .. 8 * 2304];
        const y = mlx.mlx_array_data_float32(rows[1]).?[0 .. 8 * 2304];
        for (x, y) |u, v| try testing.expectApproxEqAbs(u, v, 1e-4 * @max(1, @abs(u)));
    }
    for (0..2) |round| {
        var noise: [3]ops.Array = undefined;
        var lane_in: [3]dflash2.Drafter.BlockLane = undefined;
        var want: [3]ops.Array = undefined;
        for (&noise, &lane_in, &want, 0..) |*nz, *lane, *row, j| {
            const position = starts[j] + @as(c_int, @intCast(round)) * 3;
            nz.* = try g.concat(&.{ anchors[j], try g.broadcast(d.mask, &.{ 1, 7, 2304 }) }, 1);
            lane.* = .{ .taps = rings[j], .position = position, .cache = &batch[j] };
            row.* = try d.blockRows(&g, nz.*, rings[j], position, &alone[j]);
        }
        const got = try d.blockRowsBatch(&g, try g.concat(&noise, 0), &lane_in);
        for (want, 0..) |x, j| {
            const at: c_int = @intCast(j);
            const y = try g.sliceAxis(got, 0, at, at + 1);
            const xf = try g.astype(x, .float32);
            const yf = try g.astype(y, .float32);
            try ops.evalMany(&.{ xf, yf });
            const xs = mlx.mlx_array_data_float32(xf).?[0 .. 8 * 2304];
            const ys = mlx.mlx_array_data_float32(yf).?[0 .. 8 * 2304];
            var dot: f64 = 0;
            var xx: f64 = 0;
            var yy: f64 = 0;
            for (xs, ys) |u, v| {
                dot += @as(f64, u) * v;
                xx += @as(f64, u) * u;
                yy += @as(f64, v) * v;
            }
            // Other matmul tiles round differently (cos 0.996+ on NAX, 0.96+ on the M4 Max): a lane is its own, not its alone's bytes.
            try testing.expect(dot / @sqrt(xx * yy) > 0.95);
            try testing.expect(@sqrt(xx / yy) > 0.95 and @sqrt(xx / yy) < 1.05);
        }
    }
    // Drawn: each lane's law sums to one over its candidates and holds the token drawn from it.
    var lanes: [3]dflash2.Drafter.DraftLane = undefined;
    for (&lanes, 0..) |*lane, j| lane.* = .{ .t1 = t1[j], .anchor = anchors[j], .taps = rings[j], .position = starts[j] + 6, .cache = &batch[j], .steps = @intCast(3 + 2 * j) };
    var got: [3]draft.Drafter.Block = undefined;
    try d.draftBatch(&g, &lanes, .{ .temperatures = &.{ 1.0, 0.8, 1.0 }, .seed = 5 }, &w.head.lin, w.head.vocab, &got);
    for (got, 0..) |blk, j| {
        const steps: usize = 3 + 2 * j;
        try ops.evalMany(&.{ blk.tokens, blk.probs });
        const toks = mlx.mlx_array_data_int32(blk.tokens).?[0..steps];
        const law = mlx.mlx_array_data_float32(blk.probs).?;
        const vocab: usize = @intCast(w.head.vocab);
        for (0..steps) |r| {
            var sum: f32 = 0;
            for (law[r * vocab ..][0..vocab]) |x| sum += x;
            try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-3);
            try testing.expect(law[r * vocab + @as(usize, @intCast(toks[r]))] > 0);
        }
    }
    for (&alone, &batch) |*x, *y| {
        x.release();
        y.release();
    }
}

// Bar: `relsplit_lanes.metal` (non-NAX M3/M4 4-lane, 2..16-row single-pass kernel) matches `splitRead` across
// sliding & global layers, direct & multi-split, 1..4 lanes, and 2..16 rows (both NR=1 and NR=2).
test "inkling2 kernels: relsplit_lanes (M4) matches splitRead across lanes and rows" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const heads: c_int = 24;
    const kv_heads: c_int = 8;
    const hd: c_int = 128;
    const d_rel: c_int = 16;
    const r_ext: c_int = 1024;
    var cfg: plugin.config.Config = undefined;
    cfg.hidden_size = 2304;
    cfg.num_attention_heads = @intCast(heads);
    cfg.num_key_value_heads = @intCast(kv_heads);
    cfg.head_dim = @intCast(hd);
    cfg.d_rel = @intCast(d_rel);
    cfg.sliding_window = 1024;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.2;
    const cached_at = [_]c_int{ 40, 700, 1500, 130 };
    var seed: u64 = 81000;
    for ([_]bool{ false, true }) |global| for ([_]c_int{ 2, 8, 12, 16 }) |rows| for ([_]usize{ 1, 4 }) |n| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const dt: ops.Dtype = .bfloat16;
        var L: layers.Layer = undefined;
        L.global = global;
        L.rel_proj = try randomRow(&g, a, seed, &.{ d_rel, r_ext }, 0.3, dt);
        const total: c_int = rows * @as(c_int, @intCast(n));
        const q = try randomRow(&g, a, seed + 1, &.{ 1, total, heads, hd }, 0.6, dt);
        const rel_st = try randomRow(&g, a, seed + 2, &.{ 1, total, heads, d_rel }, 0.5, dt);
        const sinks = try randomRow(&g, a, seed + 3, &.{ 1, total, heads }, 1.0, dt);
        const kb = try randomRow(&g, a, seed + 6, &.{ 1, kv_heads, total, hd }, 0.6, dt);
        const vb = try randomRow(&g, a, seed + 7, &.{ 1, kv_heads, total, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 8, &.{total * heads * hd}, 2.0, .float32));
        var ks: [4]ops.Array = undefined;
        var vs: [4]ops.Array = undefined;
        var offsets: [4]c_int = undefined;
        var want: [4]ops.Array = undefined;
        for (0..n) |i| {
            const cached = cached_at[i];
            ks[i] = try randomRow(&g, a, seed + 30 + i, &.{ 1, kv_heads, cached, hd }, 0.6, dt);
            vs[i] = try randomRow(&g, a, seed + 40 + i, &.{ 1, kv_heads, cached, hd }, 1.0, dt);
            offsets[i] = cached;
            const at: c_int = @as(c_int, @intCast(i)) * rows;
            const from: c_int = if (global) 0 else @max(0, cached - 1023);
            want[i] = (try kernels.splitRead(&g, &L, &cfg, try g.sliceAxis(q, 1, at, at + rows), try g.sliceAxis(rel_st, 1, at, at + rows), try g.sliceAxis(sinks, 1, at, at + rows), try g.sliceAxis(ks[i], 2, from, cached), try g.sliceAxis(vs[i], 2, from, cached), .{ try g.sliceAxis(kb, 2, at, at + rows), try g.sliceAxis(vb, 2, at, at + rows) }, cached, try g.sliceAxis(gate, 0, at * heads * hd, (at + rows) * heads * hd))) orelse return error.TestUnexpectedResult;
        }
        const got = (try kernels.splitReadLanesM4(&g, &L, &cfg, q, rel_st, sinks, ks[0..n], vs[0..n], .{ kb, vb }, offsets[0..n], gate, rows, null)) orelse return error.TestUnexpectedResult;
        try expectClose(got, if (n == 1) want[0] else try g.concat(want[0..n], 1), 6e-2, "relsplit_lanes M4");
    };
}

// Bar: `relsplit_lanes.metal` with `TREE=1` matches exact per-branch `splitRead` at EVERY node of the 8-node
// branching tree `{-1, 0, 0, 1, 1, 2, 4, 5}` (sliding + global), isolating tree attention from fixture router ties.
test "inkling2 kernels: relsplit_lanes TREE=1 matches per-branch splitRead at every node" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    const heads: c_int = 24;
    const kv_heads: c_int = 8;
    const hd: c_int = 128;
    const d_rel: c_int = 16;
    const r_ext: c_int = 1024;
    var cfg: plugin.config.Config = undefined;
    cfg.hidden_size = 2304;
    cfg.num_attention_heads = @intCast(heads);
    cfg.num_key_value_heads = @intCast(kv_heads);
    cfg.head_dim = @intCast(hd);
    cfg.d_rel = @intCast(d_rel);
    cfg.sliding_window = 1024;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.2;
    const W: usize = 8;
    const par = [W]i32{ -1, 0, 0, 1, 1, 2, 4, 5 };
    var meta: [W * 2]i32 = undefined;
    for (0..W) |r| {
        if (r == 0) {
            meta[0] = 0;
            meta[1] = 1;
        } else {
            const p: usize = @intCast(par[r]);
            meta[r * 2 + 0] = meta[p * 2 + 0] + 1;
            meta[r * 2 + 1] = meta[p * 2 + 1] | (@as(i32, 1) << @intCast(r));
        }
    }
    var seed: u64 = 82500;
    for ([_]bool{ false, true }) |global| for ([_]c_int{ 40, 700 }) |cached| {
        var g = ops.Graph.init(a, fx_stream);
        defer g.deinit();
        seed += 20;
        const dt: ops.Dtype = .bfloat16;
        var L: layers.Layer = undefined;
        L.global = global;
        L.rel_proj = try randomRow(&g, a, seed, &.{ d_rel, r_ext }, 0.3, dt);
        const q = try randomRow(&g, a, seed + 1, &.{ 1, W, heads, hd }, 0.6, dt);
        const rel_st = try randomRow(&g, a, seed + 2, &.{ 1, W, heads, d_rel }, 0.5, dt);
        const sinks = try randomRow(&g, a, seed + 3, &.{ 1, W, heads }, 1.0, dt);
        const k = try randomRow(&g, a, seed + 4, &.{ 1, kv_heads, cached, hd }, 0.6, dt);
        const v = try randomRow(&g, a, seed + 5, &.{ 1, kv_heads, cached, hd }, 1.0, dt);
        const kb = try randomRow(&g, a, seed + 6, &.{ 1, kv_heads, W, hd }, 0.6, dt);
        const vb = try randomRow(&g, a, seed + 7, &.{ 1, kv_heads, W, hd }, 1.0, dt);
        const gate = try g.sigmoid(try randomRow(&g, a, seed + 8, &.{W * heads * hd}, 2.0, .float32));
        const tmeta = try g.fromSlice(i32, &meta, &.{ W, 2 });
        const got = (try kernels.splitReadLanesM4(&g, &L, &cfg, q, rel_st, sinks, &.{k}, &.{v}, .{ kb, vb }, &.{cached}, gate, W, tmeta)) orelse return error.TestUnexpectedResult;
        const from: c_int = if (global) 0 else @max(0, cached - 1023);
        const kc = try g.sliceAxis(k, 2, from, cached);
        const vc = try g.sliceAxis(v, 2, from, cached);
        for (1..W) |node| {
            var chain: [W]i32 = undefined;
            var len: usize = 0;
            var cur: i32 = @intCast(node);
            while (cur >= 0) {
                chain[len] = cur;
                len += 1;
                cur = par[@intCast(cur)];
            }
            std.mem.reverse(i32, chain[0..len]);
            const idx = try g.fromSlice(i32, chain[0..len], &.{@intCast(len)});
            const q_c = try g.take(q, idx, 1);
            const rel_c = try g.take(rel_st, idx, 1);
            const sink_c = try g.take(sinks, idx, 1);
            const kb_c = try g.take(kb, idx, 2);
            const vb_c = try g.take(vb, idx, 2);
            const gate_c = try g.reshape(try g.take(try g.reshape(gate, &.{ W, heads * hd }), idx, 0), &.{@as(c_int, @intCast(len)) * heads * hd});
            const want_c = (try kernels.splitRead(&g, &L, &cfg, q_c, rel_c, sink_c, kc, vc, .{ kb_c, vb_c }, cached, gate_c)) orelse return error.TestUnexpectedResult;
            const last: c_int = @intCast(len - 1);
            const want_row = try g.sliceAxis(want_c, 1, last, last + 1);
            const got_row = try g.sliceAxis(got, 1, @intCast(node), @intCast(node + 1));
            try expectClose(got_row, want_row, 6e-2, "relsplit_lanes TREE=1 node");
        }
    };
}

// Bar: a forward's kernel instantiations do not move with its row count.
test "inkling2 kernels: a forward of another length instantiates no kernel anew" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    const saved = kernels.prove_past;
    defer kernels.prove_past = saved;
    kernels.prove_past = 0;
    var tokens: [kernels.packed_rows + 9]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = prompt[i % prompt.len];
    var counts: [3]usize = undefined;
    for ([_]usize{ tokens.len - 2, tokens.len - 1, tokens.len }, &counts) |n, *count| {
        var cache = try test_cache.TestCache.init(a, s.pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        a.free(try step(a, &s.pack, &cache, tokens[0..n], 0, .last_row));
        count.* = kernels.instantiations();
    }
    try testing.expect(counts[0] > 0);
    try testing.expectEqual(counts[0], counts[1]);
    try testing.expectEqual(counts[1], counts[2]);
}

// Bar: a branching draft tree verified in one forward (with `Open.tree`) produces the same logits at every node as
// a fresh forward of that node's ancestor path, and committing a non-linear branch via `commitPath` leaves the cache
// and conv windows identical to serial forward along that branch.
test "inkling2 proof: branching tree verify and commitPath match serial forward along every branch" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var s = try Setup.init(a, true);
    defer s.deinit(a);
    const m: *plugin.Loaded = @ptrCast(@alignCast(s.pack.model));
    const run_mod = @import("../../sdk/bases/run.zig");
    const visits = s.pack.parsed.shell.num_hidden_layers;
    const split: usize = 8;

    // 8-node branching tree:
    //   0 (root) -> 1, 2
    //   1 -> 3, 4
    //   2 -> 5
    //   4 -> 6
    //   5 -> 7
    const W: usize = 8;
    const tree_toks = [W]u32{ prompt[split], 10, 20, 11, 12, 21, 13, 22 };
    const tree_par = [W]i32{ -1, 0, 0, 1, 1, 2, 4, 5 };
    var tree_meta: [W * 2]i32 = undefined;
    for (0..W) |r| {
        if (r == 0) {
            tree_meta[0] = 0;
            tree_meta[1] = 1;
        } else {
            const p: usize = @intCast(tree_par[r]);
            tree_meta[r * 2 + 0] = tree_meta[p * 2 + 0] + 1;
            tree_meta[r * 2 + 1] = tree_meta[p * 2 + 1] | (@as(i32, 1) << @intCast(r));
        }
    }

    var cache = try test_cache.TestCache.init(a, visits);
    defer cache.deinit();
    a.free(try step(a, &s.pack, &cache, prompt[0..split], 0, .last_row));

    var g = ops.Graph.init(a, m.base.s);
    defer g.deinit();
    const opens = try a.alloc(kernels.Open, visits);
    defer a.free(opens);
    const tctx: kernels.TreeCtx = .{
        .parent = try g.fromSlice(i32, &tree_par, &.{ 1, @intCast(W) }),
        .meta = try g.fromSlice(i32, &tree_meta, &.{ @intCast(W), 2 }),
    };
    for (opens) |*o| o.* = .{ .g = &g, .tree = tctx };
    var sp: plugin.Spec = .{ .visits = &.{}, .keep = 4 };
    defer {
        m.base.weights.spec = null;
        for ([_]mlx.mlx_array{ sp.ring, sp.held }) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
    }
    m.base.weights.spec = &sp;
    sp.opens = opens;
    const tree_logits = try step(a, &s.pack, &cache, &tree_toks, split, .all_rows);
    defer a.free(tree_logits);
    sp.opens = null;

    // Every tree node's logits must match serial forward along its ancestor chain.
    for (0..W) |node| {
        var chain: [W]u32 = undefined;
        var len: usize = 0;
        var cur: i32 = @intCast(node);
        while (cur >= 0) {
            chain[len] = tree_toks[@intCast(cur)];
            len += 1;
            cur = tree_par[@intCast(cur)];
        }
        std.mem.reverse(u32, chain[0..len]);
        var serial = try test_cache.TestCache.init(a, visits);
        defer serial.deinit();
        a.free(try step(a, &s.pack, &serial, prompt[0..split], 0, .last_row));
        const want = try step(a, &s.pack, &serial, chain[0..len], split, .last_row);
        defer a.free(want);
        const got_row = tree_logits[node * VOCAB .. (node + 1) * VOCAB];
        try testing.expect(parity.qwen4CompareRows(got_row, want, 1, VOCAB).min_cos > 0.999);
    }

    // Commit non-linear branch {0, 2, 5, 7} and verify the next step matches serial forward along {0, 2, 5, 7}.
    const branch = [_]u32{ 0, 2, 5, 7 };
    const branch_toks = [_]u32{ tree_toks[0], tree_toks[2], tree_toks[5], tree_toks[7] };
    const follow = [_]u32{33};
    var h = [_]abi.Handle{cache.handle()};
    const pos = [_]u32{split};
    const cr: run_mod.Run = .{ .mode = .verify, .rows = @intCast(branch.len), .positions = &pos, .caches = &h, .cache_ops = &test_cache.ops };
    var cc = cr.ctx(&g, &m.base.shared);
    try kernels.commitPath(&cc, &m.base.weights.body.cfg, opens, &branch, 0, 1);
    try testing.expectEqual(@as(u64, split + branch.len), test_cache.ops.len(cache.handle(), 0));
    const got_next = try step(a, &s.pack, &cache, &follow, split + branch.len, .last_row);
    defer a.free(got_next);

    var serial = try test_cache.TestCache.init(a, visits);
    defer serial.deinit();
    a.free(try step(a, &s.pack, &serial, prompt[0..split], 0, .last_row));
    a.free(try step(a, &s.pack, &serial, &branch_toks, split, .last_row));
    const want_next = try step(a, &s.pack, &serial, &follow, split + branch.len, .last_row);
    defer a.free(want_next);
    try testing.expect(parity.qwen4CompareRows(got_next, want_next, 1, VOCAB).min_cos > 0.999);
}

// Bar: `sampleAndAcceptTree` matches greedy argmax (with suppress mask, stops, non-linear tree walk), enforces
// top-k/top-p/min-p support cutoffs, and preserves the exact target distribution under multi-child without-replacement
// sparse-16 rejection sampling.
test "inkling2 kernels: sampleAndAcceptTree greedy, filtering, and exact multi-child distribution" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    fx_stream = mlx.gpuStream();
    var g = ops.Graph.init(a, fx_stream);
    defer g.deinit();

    // 1. Greedy tree walk with suppress mask and non-linear branch {0 -> 2 -> 4}.
    {
        const V: c_int = 64;
        const W: c_int = 6;
        var logits = try a.alloc(f32, @intCast(W * V));
        defer a.free(logits);
        @memset(logits, -5.0);
        // Row 0 prefers token 7 (suppressed!) and second-best token 20 (child 2).
        logits[0 * V + 7] = 10.0;
        logits[0 * V + 20] = 8.0;
        // Row 2 prefers token 30 (child 4).
        logits[2 * V + 30] = 9.0;
        // Row 4 prefers token 45 (not a child -> becomes next_token).
        logits[4 * V + 45] = 9.0;
        var supp: [64]u8 = @splat(0);
        supp[7] = 1;
        const toks = try g.fromSlice(i32, &.{ 5, 10, 20, 11, 30, 31 }, &.{ 1, W });
        const pars = try g.fromSlice(i32, &.{ -1, 0, 0, 1, 2, 2 }, &.{ 1, W });
        const res = try kernels.sampleAndAcceptTree(
            &g,
            try g.fromSlice(f32, logits, &.{ W, V }),
            V,
            try g.adopt(mlx.mlx_array_new_data(&supp, &[_]c_int{V}, 1, .bool_)),
            toks,
            pars,
            &.{.{ .temperature = 0.0, .accept_cap = 5 }},
            W,
            &.{2},
            null,
        );
        try ops.eval(res);
        const r = mlx.mlx_array_data_int32(res).?;
        try testing.expectEqual(@as(i32, 2), r[0]); // accepted = 2
        try testing.expectEqual(@as(i32, 45), r[1]); // next_token = 45
        try testing.expectEqualSlices(i32, &.{ 0, 2, 4 }, r[2 .. 2 + 3]);
        try testing.expectEqualSlices(i32, &.{ 5, 20, 30 }, r[2 + W .. 2 + W + 3]);
    }

    // 2. Statistical exactness of 4-child without-replacement sparse-16 rejection sampling on a 32-token vocab.
    {
        const V: usize = 32;
        const K: usize = 16;
        const W: c_int = 5;
        const temp: f32 = 0.85;
        var target_logits: [V]f32 = undefined;
        var target_p: [V]f32 = undefined;
        var max_l: f32 = -std.math.inf(f32);
        for (0..V) |v| {
            target_logits[v] = @as(f32, @floatFromInt(@as(i32, @intCast((v * 7) % 13)) - 6)) * 0.4;
            max_l = @max(max_l, target_logits[v] / temp);
        }
        var sum_p: f32 = 0;
        for (0..V) |v| {
            target_p[v] = @exp(target_logits[v] / temp - max_l);
            sum_p += target_p[v];
        }
        for (0..V) |v| target_p[v] /= sum_p;

        // Deliberately mismatched draft proposal over first 16 tokens, 4 children placed at root: {0, 1, 2, 3}.
        var q_row: [K]f32 = undefined;
        var q_sum: f32 = 0;
        for (0..K) |k| {
            q_row[k] = 1.0 / @as(f32, @floatFromInt(k + 2));
            q_sum += q_row[k];
        }
        for (0..K) |k| q_row[k] /= q_sum;

        // Batch 16 lanes x 200 batches = 3200 independent draws at root row 0.
        const B: usize = 16;
        var counts: [V]u32 = @splat(0);
        const total_draws: usize = B * 200;

        var lg_buf = try a.alloc(f32, B * @as(usize, @intCast(W)) * V);
        defer a.free(lg_buf);
        for (0..B * @as(usize, @intCast(W))) |r| @memcpy(lg_buf[r * V ..][0..V], &target_logits);
        const logits_arr = try g.fromSlice(f32, lg_buf, &.{ @intCast(B * @as(usize, @intCast(W))), @intCast(V) });

        var tok_buf: [B * 5]i32 = undefined;
        var par_buf: [B * 5]i32 = undefined;
        var meta_buf: [B * 5 * 2]i32 = @splat(0);
        var cand_buf: [B * 1 * K]i32 = undefined;
        var nq_buf: [B * 5 * K]f32 = @splat(0);
        var draw_buf: [B * 5 * 4]u8 = @splat(255);
        for (0..B) |b| {
            par_buf[b * 5 + 0] = -1;
            par_buf[b * 5 + 1] = 0;
            par_buf[b * 5 + 2] = 0;
            par_buf[b * 5 + 3] = 0;
            par_buf[b * 5 + 4] = 0;
            for (0..K) |k| {
                cand_buf[b * K + k] = @intCast(k);
                nq_buf[(b * 5 + 0) * K + k] = q_row[k];
            }
        }
        const pars_arr = try g.fromSlice(i32, &par_buf, &.{ @intCast(B), W });
        const meta_arr = try g.fromSlice(i32, &meta_buf, &.{ @intCast(B * 5), 2 });
        const cand_arr = try g.fromSlice(i32, &cand_buf, &.{ @intCast(B), 1, @intCast(K) });
        const nq_arr = try g.fromSlice(f32, &nq_buf, &.{ @intCast(B), W, @intCast(K) });
        var prng = std.Random.DefaultPrng.init(0x5a3b);
        const rng = prng.random();
        for (0..200) |iter| {
            var sl: [B]kernels.SampleLane = undefined;
            for (0..B) |b| {
                // Draw 4 distinct candidates without replacement from q_row (exact without-replacement proposal).
                var used: [K]bool = @splat(false);
                var rem: f32 = 1.0;
                tok_buf[b * 5 + 0] = 0;
                for (0..4) |ch| {
                    var u = rng.float(f32) * rem;
                    var pick: usize = K - 1;
                    for (0..K) |k| {
                        if (used[k]) continue;
                        u -= q_row[k];
                        if (u <= 0) {
                            pick = k;
                            break;
                        }
                    }
                    used[pick] = true;
                    rem -= q_row[pick];
                    tok_buf[b * 5 + 1 + ch] = @intCast(pick);
                    draw_buf[(b * 5 + 0) * 4 + ch] = @intCast(pick);
                }
                sl[b] = .{ .temperature = temp, .top_k = 0, .top_p = 1.0, .min_p = 0.0, .seed = @as(u64, iter * B + b) *% 0x9E3779B97F4A7C15 + 1, .accept_cap = 1 };
            }
            var ig = g.scope();
            defer ig.deinit();
            const ti: kernels.SampleTreeIn = .{
                .tmeta = meta_arr,
                .cands = cand_arr,
                .next_q = nq_arr,
                .draws = try ig.adopt(mlx.mlx_array_new_data(&draw_buf, &[_]c_int{ @intCast(B), W, 4 }, 3, .uint8)),
                .r = 1,
                .k = @intCast(K),
            };
            const res = try kernels.sampleAndAcceptTree(&ig, logits_arr, @intCast(V), .{}, try ig.fromSlice(i32, &tok_buf, &.{ @intCast(B), W }), pars_arr, &sl, W, &.{}, ti);
            try ops.eval(res);
            const rdata = mlx.mlx_array_data_int32(res).?;
            const stride = 2 + 2 * @as(usize, @intCast(W));
            for (0..B) |b| {
                const row = rdata[b * stride .. (b + 1) * stride];
                // Root selection is emitted_tokens[1] if accepted == 1 else next_token.
                const drawn: usize = @intCast(if (row[0] == 1) row[2 + @as(usize, @intCast(W)) + 1] else row[1]);
                counts[drawn] += 1;
            }
        }
        var max_err: f32 = 0;
        for (0..V) |v| {
            const emp = @as(f32, @floatFromInt(counts[v])) / @as(f32, @floatFromInt(total_draws));
            max_err = @max(max_err, @abs(emp - target_p[v]));
        }
        try testing.expect(max_err < 0.025);
    }
}

// Bar: the bundled head's groups run up to 8 slots on NAX (2 elsewhere); DFlash2's run to the GPU's width.
test "inkling2 plugin: a group of the bundled head's slots is 8 wide on NAX (2 elsewhere), of DFlash2's the GPU's" {
    try testing.expectEqual(@as(u32, 8), plugin.groupWidth(false, true));
    try testing.expectEqual(@as(u32, 2), plugin.groupWidth(false, false));
    try testing.expectEqual(@as(u32, 16), plugin.groupWidth(true, true));
    try testing.expectEqual(@as(u32, 8), plugin.groupWidth(true, false));
}

test "inkling2 plugin: a lane's tree rows follow the device, the lanes and the sampling" {
    const nax = @import("../../mlx/device.zig").naxAvailable();
    try testing.expectEqual(@as(u32, 8), plugin.treeWidth(1, false));
    try testing.expectEqual(@as(u32, if (nax) 8 else 5), plugin.treeWidth(2, false));
    try testing.expectEqual(@as(u32, if (nax) 6 else 5), plugin.treeWidth(4, false));
    try testing.expectEqual(@as(u32, if (nax) 6 else 4), plugin.treeWidth(16, false));
    try testing.expectEqual(@as(u32, if (nax) 6 else 8), plugin.treeWidth(1, true));
    try testing.expectEqual(@as(u32, 5), plugin.treeWidth(4, true));
    try testing.expectEqual(@as(u32, 4), plugin.treeWidth(8, true));
}

/// The command-buffer pair a fake host holds.
const Limits = struct {
    var op_split: i32 = 50;
    var mb_split: i32 = 50;
    fn set(_: ?*anyopaque, o: i32, m: i32, prev_ops: *i32, prev_mb: *i32) callconv(.c) @import("../../abi/host.zig").Verdict {
        prev_ops.* = op_split;
        prev_mb.* = mb_split;
        if (o > 0) op_split = o;
        if (m > 0) mb_split = m;
        return .ok;
    }
};

// Bar: a one-row forward runs under the plugin's command-buffer MB split and a wider one under the host's, the op split
// is never moved, the unload puts the host's pair back, and a load that fails never moves it.
test "inkling2 load: decode steps lift the command-buffer MB split, prompts and the unload restore the host's" {
    const a = testing.allocator;
    var fake: @import("../../sdk/testing/fake_host.zig").FakeHost = undefined;
    fake.init(.{});
    fake.gpu_table.command_buffer_limits = Limits.set;
    Limits.op_split = 50;
    Limits.mb_split = 50;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx = try plugin.conformanceFixture(arena.allocator());

    var pack = try conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, fx, .{ .host = fake.host() });
    try testing.expectEqual(@as(i32, 50), Limits.mb_split);
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    a.free(try step(a, &pack, &cache, prompt[0..8], 0, .last_row));
    try testing.expectEqual(@as(i32, 50), Limits.mb_split);
    a.free(try step(a, &pack, &cache, prompt[8..9], 8, .last_row));
    try testing.expectEqual(@as(i32, plugin.command_buffer_mb), Limits.mb_split);
    a.free(try step(a, &pack, &cache, prompt[9..13], 9, .last_row));
    try testing.expectEqual(@as(i32, 50), Limits.mb_split);
    a.free(try step(a, &pack, &cache, prompt[13..14], 13, .last_row));
    try testing.expectEqual(@as(i32, plugin.command_buffer_mb), Limits.mb_split);
    pack.close();
    try testing.expectEqual(@as(i32, 50), Limits.mb_split);
    try testing.expectEqual(@as(i32, 50), Limits.op_split);

    // The embedding renamed in the header: the parse passes and the plugin's own load fails.
    const files = try arena.allocator().dupe(fixture.File, fx.files);
    for (files) |*f| if (std.mem.endsWith(u8, f.name, ".safetensors")) {
        const bytes = try arena.allocator().dupe(u8, f.bytes);
        const at = std.mem.indexOf(u8, bytes, "embed_tokens.weight") orelse continue;
        bytes[at] = 'X';
        f.bytes = bytes;
    };
    Limits.mb_split = 77;
    if (conformance.Loaded.openWith(testing.io, arena.allocator(), plugin.plugin.name, &plugin.table, .{ .files = files }, .{ .host = fake.host() })) |loaded| {
        var p = loaded;
        p.close();
        return error.TestUnexpectedResult;
    } else |_| {}
    try testing.expectEqual(@as(i32, 77), Limits.mb_split);
    try testing.expectEqual(@as(i32, 50), Limits.op_split);
}

// Bar: `draftBatchPacked` matches serial `draft()` across 4 lanes on GPU (bf16 fused Markov path) with carried
// `ContextKv` caches and different positions, and produces valid normalized distributions in sampled mode.
test "inkling2 draft: draftBatchPacked matches serial draft across lanes (greedy and sampled)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const saved = fx_stream;
    fx_stream = mlx.gpuStream();
    defer fx_stream = saved;
    var g = ops.Graph.init(a, fx_stream);
    defer g.deinit();
    const H = 64;
    const HD = 16;
    const V = 64;
    const R = 16;
    const dt: ops.Dtype = .bfloat16;
    var seed: u64 = 93000;
    const W = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, shape: []const c_int, scale: f32) !ops.Array {
            sd.* += 1;
            return randomRow(gr, al, sd.*, shape, scale, dt);
        }
    };
    const lin = struct {
        fn of(gr: *ops.Graph, al: std.mem.Allocator, sd: *u64, out: c_int, in: c_int) !layers.Linear {
            return .{ .w = try W.of(gr, al, sd, &.{ out, in }, 0.15) };
        }
    };
    const one = try g.ones(&.{H}, dt);
    const one32 = try g.ones(&.{H}, .float32);
    var layer: draft.Layer = .{
        .in_norm = one,
        .post_norm = one,
        .mix_norm = one32,
        .q = try lin.of(&g, a, &seed, 2 * HD, H),
        .k = try lin.of(&g, a, &seed, HD, H),
        .v = try lin.of(&g, a, &seed, HD, H),
        .o = try lin.of(&g, a, &seed, H, 2 * HD),
        .k_ctx = try lin.of(&g, a, &seed, HD, H),
        .v_ctx = try lin.of(&g, a, &seed, HD, H),
        .q_norm = try g.ones(&.{HD}, dt),
        .k_norm = try g.ones(&.{HD}, dt),
        .k_norm_ctx = try g.ones(&.{HD}, dt),
        .gate = try lin.of(&g, a, &seed, 96, H),
        .up = try lin.of(&g, a, &seed, 96, H),
        .down = try lin.of(&g, a, &seed, H, 96),
    };
    layer.qkv = .{ .w = try g.concat(&.{ layer.q.w, layer.k.w, layer.v.w }, 0) };
    layer.kv_ctx = .{ .w = try g.concat(&.{ layer.k_ctx.w, layer.v_ctx.w }, 0) };
    layer.gate_up = .{ .w = try g.concat(&.{ layer.gate.w, layer.up.w }, 0) };
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 1, .heads = 2, .kv_heads = 1, .head_dim = HD, .intermediate = 96, .window = 16, .block = 5, .rank = R, .tap_count = 2 },
        .hidden = H,
        .s = fx_stream,
    };
    d.layers = (&layer)[0..1];
    d.norm = one;
    d.mask = try W.of(&g, a, &seed, &.{ 1, 1, H }, 0.8);
    d.alpha = try g.softmax(try randomRow(&g, a, seed + 1, &.{ 2, 2, 1 }, 1.0, .float32), 1, true);
    d.prev_rows = try W.of(&g, a, &seed, &.{ V, R }, 0.4);
    d.rnn = .{ .w = try W.of(&g, a, &seed, &.{ 3 * R, 2 * R + 2 * H }, 0.08) };
    d.rnn_bias = try W.of(&g, a, &seed, &.{3 * R}, 0.08);
    d.rnn_norm = one32;
    d.fold = try W.of(&g, a, &seed, &.{ R, V }, 0.25);
    d.fold_bias = try W.of(&g, a, &seed, &.{ 1, 1, V }, 0.25);
    const head = try lin.of(&g, a, &seed, V, H);

    const B = 4;
    var anchors: [B]ops.Array = undefined;
    var t1s: [B]ops.Array = undefined;
    var fulls: [B]ops.Array = undefined;
    var batch_ctxs: [B]draft.ContextKv = @splat(.{});
    var solo_ctxs: [B]draft.ContextKv = @splat(.{});
    defer for (&batch_ctxs, &solo_ctxs) |*bc, *sc| {
        bc.release();
        sc.release();
    };
    for (0..B) |j| {
        anchors[j] = try W.of(&g, a, &seed, &.{ 1, 1, H }, 1.0);
        t1s[j] = try g.fromSlice(i32, &.{@intCast(3 + j * 5)}, &.{1});
        fulls[j] = try W.of(&g, a, &seed, &.{ 1, 24, 2 * H }, 1.0);
    }
    const spans = [_][2]c_int{ .{ 0, 6 }, .{ 0, 10 }, .{ 4, 16 } };
    for (spans) |sp| {
        var lanes: [B]draft.Drafter.DraftLane = undefined;
        for (0..B) |j| {
            const hi = sp[1] + @as(c_int, @intCast(j));
            const lo = @max(0, hi - 15);
            lanes[j] = .{
                .t1 = t1s[j],
                .anchor = anchors[j],
                .taps = try g.sliceAxis(fulls[j], 1, lo, hi),
                .position = hi,
                .cache = &batch_ctxs[j],
                .steps = 4,
            };
        }
        const got = try d.draftBatchPacked(&g, &lanes, &head, V - 4, 30, null);
        try ops.evalMany(&.{ got.tokens, got.confidence });
        const got_t = mlx.mlx_array_data_int32(got.tokens).?;
        const got_c = mlx.mlx_array_data_float32(got.confidence).?;
        for (0..B) |j| {
            const want = try d.draft(&g, lanes[j].t1, lanes[j].anchor, lanes[j].taps, lanes[j].position, &head, V - 4, 30, &solo_ctxs[j], null);
            try ops.evalMany(&.{ want.tokens, want.confidence });
            const wt = mlx.mlx_array_data_int32(want.tokens).?;
            const wc = mlx.mlx_array_data_float32(want.confidence).?;
            for (0..4) |k| {
                try testing.expectEqual(wt[k], got_t[j * 4 + k]);
                try testing.expectApproxEqAbs(wc[k], got_c[j * 4 + k], 5e-2);
            }
        }
    }
    // Sampled path: probabilities normalize to 1 and each drawn token has positive probability.
    var slanes: [B]draft.Drafter.DraftLane = undefined;
    for (0..B) |j| slanes[j] = .{ .t1 = t1s[j], .anchor = anchors[j], .taps = try g.sliceAxis(fulls[j], 1, 0, 12), .position = 12, .cache = &batch_ctxs[j], .steps = 3 };
    var blocks: [B]draft.Drafter.Block = undefined;
    try d.draftBatch(&g, &slanes, &head, V - 4, 30, .{ .temperatures = &.{ 0.8, 1.0, 0.7, 1.2 }, .seed = 99 }, &blocks);
    const vocab: usize = V - 4;
    for (blocks) |blk| {
        try ops.evalMany(&.{ blk.tokens, blk.probs });
        const toks = mlx.mlx_array_data_int32(blk.tokens).?[0..3];
        const law = mlx.mlx_array_data_float32(blk.probs).?;
        for (0..3) |r| {
            var sum: f32 = 0;
            for (law[r * vocab ..][0..vocab]) |x| sum += x;
            try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-3);
            try testing.expect(law[r * vocab + @as(usize, @intCast(toks[r]))] > 0);
        }
    }
    // GPU-fused sparse top-K sampled path: normalized proposal probs, drawn token matches candidate, mixed greedy+sampled.
    if (kernels.markovRuns(&g, d.rnn.w, d.fold, R)) {
        const s_params = [B]kernels.SampleLane{
            .{ .temperature = 0.0 },
            .{ .temperature = 1.0, .top_p = 0.95 },
            .{ .temperature = 0.8, .top_k = 8 },
            .{ .temperature = 1.2, .min_p = 0.05 },
        };
        const sb = try d.draftBatchSampled(&g, &slanes, &s_params, &.{ 11, 22, 33, 44 }, &head, V - 4, 30);
        try ops.evalMany(&.{ sb.tokens, sb.cands, sb.probs, sb.draws });
        const stoks = mlx.mlx_array_data_int32(sb.tokens).?;
        const scands = mlx.mlx_array_data_int32(sb.cands).?;
        const sprobs = mlx.mlx_array_data_float32(sb.probs).?;
        const sdraws = mlx.mlx_array_data_uint8(sb.draws).?;
        const K: usize = @intCast(sb.k);
        for (0..B) |j| for (0..3) |r| {
            const di = sdraws[j * 3 + r];
            try testing.expect(di < K);
            const row_c = scands[(j * 3 + r) * K ..][0..K];
            const row_p = sprobs[(j * 3 + r) * K ..][0..K];
            try testing.expectEqual(row_c[di], stoks[j * 3 + r]);
            var sum: f32 = 0;
            for (row_p) |p| sum += p;
            try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-4);
            try testing.expect(row_p[di] > 0);
        };
        // Dense common-token block: 4 peaks inside the first 16 tokens must all appear in top-K for both NB=2 and NB=8.
        for ([_]c_int{ 2, 6 }) |nb_test| {
            const tv: c_int = 128;
            const p0 = try g.zeros(&.{ nb_test, 3 * R }, .float32);
            const fold0 = try g.zeros(&.{ R, tv }, .bfloat16);
            const fb0 = try g.zeros(&.{ 1, 1, tv }, .bfloat16);
            const base_buf = try a.alloc(f32, @intCast(nb_test * tv));
            defer a.free(base_buf);
            @memset(base_buf, -5.0);
            for (0..@intCast(nb_test)) |b_i| {
                base_buf[b_i * @as(usize, @intCast(tv)) + 3] = 9.0;
                base_buf[b_i * @as(usize, @intCast(tv)) + 7] = 8.5;
                base_buf[b_i * @as(usize, @intCast(tv)) + 11] = 8.0;
                base_buf[b_i * @as(usize, @intCast(tv)) + 15] = 7.5;
            }
            const base0 = try g.fromSlice(f32, base_buf, &.{ nb_test, 1, tv });
            const step0 = try g.fromSlice(i32, &.{0}, &.{1});
            const cap0 = try g.fromSlice(f32, &.{30.0}, &.{1});
            const part = try kernels.markovFoldBatchSg(&g, p0, fold0, fb0, base0, step0, tv, cap0);
            var lp6: [6 * 4]f32 = undefined;
            for (0..6) |bi| {
                lp6[bi * 4 + 0] = 1.0;
                lp6[bi * 4 + 1] = 0.0;
                lp6[bi * 4 + 2] = 0.95;
                lp6[bi * 4 + 3] = 0.0;
            }
            const lparams_arr = try g.fromSlice(f32, lp6[0..@intCast(nb_test * 4)], &.{ nb_test, 4 });
            const seeds_arr = try g.zeros(&.{ nb_test, 2 }, .uint32);
            const st0 = try g.zeros(&.{ nb_test, 1, R }, .bfloat16);
            const picked = try kernels.markovPickSample(&g, part[0], part[1], p0, st0, cap0, lparams_arr, seeds_arr, step0, kernels.markov_sample_k);
            try ops.eval(picked.cands);
            const cdata = mlx.mlx_array_data_int32(picked.cands).?;
            for (0..@intCast(nb_test)) |b_i| {
                const lane_c = cdata[b_i * K ..][0..4];
                try testing.expectEqual(@as(i32, 3), lane_c[0]);
                try testing.expectEqual(@as(i32, 7), lane_c[1]);
                try testing.expectEqual(@as(i32, 11), lane_c[2]);
                try testing.expectEqual(@as(i32, 15), lane_c[3]);
            }
        }
    }
}

// Bar: `step_group` on the bundled drafter (`stepGroupBundled`) generates the exact same greedy tokens as plain
// serial decode across 4 concurrent slots on the real Inkling-2-Mini pack.
test "inkling2 draft: stepGroupBundled matches plain decode on the release pack" {
    const env = @import("../../env.zig");
    const pack_dir = std.mem.span(env.get(.INKLING2_RELEASE) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rel = try Release.open(a, pack_dir);
    defer rel.close();
    if (rel.m.drafter == null or rel.m.drafter2 != null) return error.SkipZigTest;
    const nd = plugin.table.spec.?.native_drafter.?;
    const step_group = nd.step_group.?;
    const n_layers = rel.parsed.shell.num_hidden_layers;

    const B = 4;
    const gen_n = 16;
    const prompts = [B][]const u32{
        quokka_def[0..24],
        quokka_def[0..28],
        quokka_def[0..32],
        quokka_def[0..36],
    };

    var want_toks: [B][gen_n]u32 = undefined;
    var step_logits: [B][gen_n][]f32 = undefined;
    var first_toks: [B]u32 = undefined;
    for (0..B) |i| {
        var c = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        defer c.deinit();
        var h = [_]abi.Handle{c.handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(plugin.table.forward(rel.m, &.{ .mode = .prefill, .logits = .last_row, .rows = @intCast(prompts[i].len), .tokens = .of(prompts[i]), .slots = .of(&[_]u32{@intCast(100 + i)}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
        step_logits[i][0] = try conformance.consumeF32(a, out.logits);
        first_toks[i] = argmaxOf(step_logits[i][0]);
        var tok = first_toks[i];
        var pos: u32 = @intCast(prompts[i].len);
        for (0..gen_n) |t| {
            want_toks[i][t] = tok;
            if (t + 1 < gen_n) {
                var o2: abi.ForwardOut = .{};
                try testing.expect(plugin.table.forward(rel.m, &.{ .mode = .decode, .logits = .last_row, .rows = 1, .tokens = .of(&[_]u32{tok}), .slots = .of(&[_]u32{@intCast(100 + i)}), .positions = .of(&[_]u32{pos}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &o2).isOk());
                step_logits[i][t + 1] = try conformance.consumeF32(a, o2.logits);
                tok = argmaxOf(step_logits[i][t + 1]);
                pos += 1;
            }
        }
    }

    var caches: [B]test_cache.TestCache = undefined;
    var cur_pos: [B]u32 = undefined;
    for (0..B) |i| {
        caches[i] = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        cur_pos[i] = @intCast(prompts[i].len);
        var h = [_]abi.Handle{caches[i].handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(plugin.table.forward(rel.m, &.{ .mode = .prefill, .logits = .last_row, .rows = @intCast(prompts[i].len), .tokens = .of(prompts[i]), .slots = .of(&[_]u32{@intCast(200 + i)}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
        try ops.eval(.{ .ctx = out.logits.ctx });
        _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    }
    defer for (&caches) |*c| c.deinit();

    var cur_tok: [B]u32 = first_toks;
    var got_n: [B]usize = @splat(0);
    var got_toks: [B][gen_n]u32 = undefined;
    var rounds_run: usize = 0;
    while (rounds_run < gen_n) : (rounds_run += 1) {
        var active: [B]usize = undefined;
        var n_act: usize = 0;
        for (0..B) |i| if (got_n[i] < gen_n) {
            active[n_act] = i;
            n_act += 1;
        };
        if (n_act < 2) break;
        var fs: [B]abi.Forward = undefined;
        var outs: [B]abi.ForwardOut = @splat(.{});
        var h_arr: [B]abi.Handle = undefined;
        var s_arr: [B]u32 = undefined;
        var p_arr: [B]u32 = undefined;
        var t_arr: [B]u32 = undefined;
        var all_cap = true;
        for (0..n_act) |k| {
            const i = active[k];
            h_arr[k] = caches[i].handle();
            s_arr[k] = @intCast(200 + i);
            p_arr[k] = cur_pos[i];
            t_arr[k] = cur_tok[i];
            const rem: u32 = @intCast(gen_n - got_n[i]);
            if (rem <= 1) all_cap = false;
            fs[k] = .{
                .mode = .verify,
                .logits = .all_rows,
                .rows = 1,
                .tokens = .of(t_arr[k .. k + 1]),
                .slots = .of(s_arr[k .. k + 1]),
                .positions = .of(p_arr[k .. k + 1]),
                .caches = .of(h_arr[k .. k + 1]),
                .cache_ops = &test_cache.ops,
                .accept_cap = rem -| 1,
            };
        }
        if (!all_cap) break;
        const v = step_group(rel.m, &fs, &outs, @intCast(n_act));
        try testing.expect(v.isOk());
        for (0..n_act) |k| {
            const i = active[k];
            const rd = outs[k].round;
            try testing.expect(rd.n >= 1);
            for (0..rd.n) |j| {
                if (got_n[i] < gen_n) {
                    got_toks[i][got_n[i]] = rd.tokens[j];
                    got_n[i] += 1;
                }
            }
            cur_tok[i] = rd.next_token;
            cur_pos[i] = @intCast(rd.position);
        }
    }
    for (0..B) |i| {
        try testing.expect(got_n[i] >= 8);
        var same: usize = 0;
        while (same < got_n[i] and want_toks[i][same] == got_toks[i][same]) same += 1;
        if (same < got_n[i]) {
            try testing.expect(same >= 1);
            const lg = step_logits[i][same];
            const gap = lg[want_toks[i][same]] - lg[got_toks[i][same]];
            try testing.expect(gap <= 0.01 * @abs(lg[want_toks[i][same]]));
        }
    }
    // Sampled + mixed greedy/sampled bundled group: runs cleanly, accepts drafts, and respects valid vocab ids.
    for (0..B) |i| {
        caches[i].deinit();
        caches[i] = try test_cache.TestCache.init(std.heap.c_allocator, n_layers);
        cur_pos[i] = @intCast(prompts[i].len);
        var h = [_]abi.Handle{caches[i].handle()};
        var out: abi.ForwardOut = .{};
        try testing.expect(plugin.table.forward(rel.m, &.{ .mode = .prefill, .logits = .last_row, .rows = @intCast(prompts[i].len), .tokens = .of(prompts[i]), .slots = .of(&[_]u32{@intCast(300 + i)}), .positions = .of(&[_]u32{0}), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out).isOk());
        try ops.eval(.{ .ctx = out.logits.ctx });
        _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    }
    var scur_tok: [B]u32 = first_toks;
    var s_accepted: usize = 0;
    const vocab_size: u32 = rel.parsed.shell.vocab_size;
    for (0..4) |rd_i| {
        var fs: [B]abi.Forward = undefined;
        var outs: [B]abi.ForwardOut = @splat(.{});
        var h_arr: [B]abi.Handle = undefined;
        var s_arr: [B]u32 = undefined;
        var p_arr: [B]u32 = undefined;
        var t_arr: [B]u32 = undefined;
        for (0..B) |i| {
            h_arr[i] = caches[i].handle();
            s_arr[i] = @intCast(300 + i);
            p_arr[i] = cur_pos[i];
            t_arr[i] = scur_tok[i];
            fs[i] = .{
                .mode = .verify,
                .logits = .all_rows,
                .rows = 1,
                .tokens = .of(t_arr[i .. i + 1]),
                .slots = .of(s_arr[i .. i + 1]),
                .positions = .of(p_arr[i .. i + 1]),
                .caches = .of(h_arr[i .. i + 1]),
                .cache_ops = &test_cache.ops,
                .accept_cap = 7,
                .draft_temperature = if (i == 0) 0.0 else 1.0,
                .draft_seed = 0x12345678 +% @as(u64, @intCast(rd_i * 17 + i)),
                .top_p = 0.95,
            };
        }
        try testing.expect(step_group(rel.m, &fs, &outs, B).isOk());
        for (0..B) |i| {
            const rd = outs[i].round;
            try testing.expect(rd.n >= 1);
            try testing.expect(rd.next_token < vocab_size);
            for (0..rd.n) |j| try testing.expect(rd.tokens[j] < vocab_size);
            s_accepted += rd.accepted;
            scur_tok[i] = rd.next_token;
            cur_pos[i] = @intCast(rd.position);
        }
    }
    try testing.expect(s_accepted >= 4);
}

// Bar: when the host provides `Host.kernel` + `HostGpu.launch`, `kernels.launchGrid` builds the kernel once via the
// host and dispatches through `HostGpu.launch` with bit-identical results; when a host refuses `launch` as unsupported,
// it falls back to MLX's direct apply.
test "inkling2 kernels: HostGpu.launch cache dispatches bit-identically and falls back when unsupported" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const host_abi = @import("../../abi/host.zig");
    const FakeHost = @import("../../sdk/testing/fake_host.zig").FakeHost;
    const Hook = struct {
        var created: usize = 0;
        var launched: usize = 0;
        var made_handle: mlx.mlx_fast_metal_kernel = .{};
        fn makeKernel(_: ?*anyopaque, desc: *const host_abi.KernelDesc) callconv(.c) ?*anyopaque {
            created += 1;
            if (made_handle.ctx != null) return made_handle.ctx;
            var ins_z: [8][*:0]const u8 = undefined;
            var outs_z: [8][*:0]const u8 = undefined;
            var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
            defer arena.deinit();
            const aa = arena.allocator();
            const toZ = struct {
                fn f(al: std.mem.Allocator, bytes: []const u8) ?[*:0]const u8 {
                    const buf = al.allocSentinel(u8, bytes.len, 0) catch return null;
                    @memcpy(buf, bytes);
                    return buf.ptr;
                }
            }.f;
            for (desc.inputs.slice(), 0..) |s, i| ins_z[i] = toZ(aa, s.slice()) orelse return null;
            for (desc.outputs.slice(), 0..) |s, i| outs_z[i] = toZ(aa, s.slice()) orelse return null;
            const iv = mlx.mlx_vector_string_new_data(&ins_z, desc.inputs.len);
            defer _ = mlx.mlx_vector_string_free(iv);
            const ov = mlx.mlx_vector_string_new_data(&outs_z, desc.outputs.len);
            defer _ = mlx.mlx_vector_string_free(ov);
            const name_z = toZ(aa, desc.name.slice()) orelse return null;
            const src_z = toZ(aa, desc.source.slice()) orelse return null;
            const hdr_z = toZ(aa, desc.header.slice()) orelse return null;
            made_handle = mlx.mlx_fast_metal_kernel_new(name_z, iv, ov, src_z, hdr_z, desc.ensure_row_contiguous, desc.atomic_outputs);
            return made_handle.ctx;
        }
        fn runLaunch(_: ?*anyopaque, outputs: *host_abi.VectorArray, k: ?*anyopaque, inputs: host_abi.VectorArray, config: host_abi.KernelConfig, s_: host_abi.Stream) callconv(.c) host_abi.Verdict {
            launched += 1;
            var ov: mlx.mlx_vector_array = .{ .ctx = outputs.ctx };
            const rc = mlx.mlx_fast_metal_kernel_apply(&ov, .{ .ctx = k }, .{ .ctx = inputs.ctx }, .{ .ctx = config.ctx }, .{ .ctx = s_.ctx });
            outputs.ctx = ov.ctx;
            return if (rc == 0) .ok else host_abi.Verdict.refuse(.failed, "apply failed");
        }
    };
    Hook.created = 0;
    Hook.launched = 0;
    Hook.made_handle = .{};
    defer if (Hook.made_handle.ctx != null) {
        kernels.gated_kernel.host_src = null;
        kernels.gated_kernel.host_built = null;
        _ = mlx.mlx_fast_metal_kernel_free(Hook.made_handle);
        Hook.made_handle = .{};
    };

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const x = try randomRow(&g, a, 7100, &.{ 1, 1, 64 }, 1.0, .bfloat16);
    const gate = try g.sigmoid(try randomRow(&g, a, 7101, &.{64}, 1.0, .float32));

    // Direct baseline (no host).
    kernels.clearHost(null);
    const direct = try hostValues(a, &g, try kernels.gated(&g, x, gate, 1));
    defer a.free(direct);

    // Working Host.kernel + HostGpu.launch: kernel built once, both launches go through HostGpu.launch.
    var good: FakeHost = undefined;
    good.init(.{});
    good.table.kernel = Hook.makeKernel;
    good.gpu_table.launch = Hook.runLaunch;
    kernels.setHost(good.host());
    defer kernels.clearHost(good.host());
    const h1 = try hostValues(a, &g, try kernels.gated(&g, x, gate, 1));
    defer a.free(h1);
    const h2 = try hostValues(a, &g, try kernels.gated(&g, x, gate, 1));
    defer a.free(h2);
    try testing.expectEqual(@as(usize, 1), Hook.created);
    try testing.expectEqual(@as(usize, 2), Hook.launched);
    try testing.expectEqualSlices(f32, direct, h1);
    try testing.expectEqualSlices(f32, direct, h2);

    // Refusing FakeHost falls back cleanly to MLX's direct apply.
    var refusing: FakeHost = undefined;
    refusing.init(.{});
    kernels.setHost(refusing.host());
    defer kernels.clearHost(refusing.host());
    const fb = try hostValues(a, &g, try kernels.gated(&g, x, gate, 1));
    defer a.free(fb);
    try testing.expectEqualSlices(f32, direct, fb);
}

// Bar: non-NAX wide-tile sorted MoE (`tileScan` + `moeWide`) serves NVFP4, MXFP4, MXFP8 and affine banks across BM=32
// and BM=64, with fused soft-capped SwiGLU + RMAP gather and non-multiple-of-64 N tails, matching MLX's gather chain.
test "inkling2 kernels: moeWide serves NVFP4, MXFP4, MXFP8 and affine banks with fused SwiGLU and rmap" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable() or device.deviceGeneration().gen < 15) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();

    const E: c_int = 8;
    const S: c_int = 96;
    const top_k: c_int = 4;
    const M: c_int = S * top_k;
    const K: c_int = 128;
    const I: c_int = 64;
    const DN_OUT: c_int = 96; // non-multiple of 64 to exercise N-tail guard
    const x = try randomRow(&g, a, 7200, &.{ 1, S, K }, 0.5, .bfloat16);
    var prng = std.Random.DefaultPrng.init(7201);
    const rng = prng.random();
    const picks_host = try a.alloc(u32, @intCast(M));
    defer a.free(picks_host);
    for (picks_host) |*p| p.* = rng.uintLessThan(u32, @intCast(E));
    const flat = try g.fromSlice(u32, picks_host, &.{M});
    const order = try g.argsort(flat, 0);
    const token = try g.call(mlx.mlx_floor_divide, .{ order, try g.astype(try g.scalar(@floatFromInt(top_k), .float32), ops.dtype(order)) });
    const sel = try g.take(flat, order, 0);
    const rmap = try g.astype(token, .uint32);
    const x_sorted = try g.reshape(try g.take(try g.reshape(x, &.{ S, K }), token, 0), &.{ M, 1, K });
    const caps = try g.fromSlice(f32, &.{ 1.0 / 30.0, 30.0, 1.0 / 30.0, 30.0 }, &.{4});
    const gu_scale = try randomRow(&g, a, 7202, &.{E}, 0.5, .float32);

    const formats = [_]struct { mode: QuantMode, bits: u32, group: u32, scaled: bool }{
        .{ .mode = .nvfp4, .bits = 4, .group = 16, .scaled = true },
        .{ .mode = .mxfp4, .bits = 4, .group = 32, .scaled = false },
        .{ .mode = .mxfp8, .bits = 8, .group = 32, .scaled = false },
        .{ .mode = .affine, .bits = 4, .group = 64, .scaled = false },
        .{ .mode = .affine, .bits = 8, .group = 64, .scaled = true },
    };
    for (formats, 0..) |f, fi| {
        const gu = try packedBank(&g, a, 7210 + fi * 2, &.{ E, 2 * I, K }, f.bits, f.group, f.mode);
        const dn = try packedBank(&g, a, 7211 + fi * 2, &.{ E, DN_OUT, I }, f.bits, f.group, f.mode);
        const sc: ops.Array = if (f.scaled) gu_scale else .{};

        const gu_ref = try layers.gather(&g, x_sorted, &gu, sel, true);
        const act_ref = try kernels.activation(&g, gu_ref, sel, if (f.scaled) gu_scale else caps, caps, f.scaled);
        const dn_ref = try layers.gather(&g, act_ref, &dn, sel, true);

        for ([_]c_int{ 32, 64 }) |bm| {
            const tiles = try kernels.tileScan(&g, sel, @intCast(E), bm);
            const act_got = try kernels.moeWide(&g, x, &gu, sc, tiles, rmap, caps, true);
            const dn_got = try kernels.moeWide(&g, act_got, &dn, .{}, tiles, null, caps, false);
            try testing.expect((try parity.maxAbsDiffF32(act_got, act_ref, s)) <= 5e-2);
            try testing.expect((try parity.maxAbsDiffF32(dn_got, dn_ref, s)) <= 5e-2);
        }
    }
}

// Bar: `relativeWith` with a fused output gate matches `gated(relative(...))` bit-for-bit on both global layers past
// distance R (exercising the plain tile loop) and sliding-window layers.
test "inkling2 kernels: relativeWith fuses output gate and runs global plain loop past R" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();

    var cfg = std.mem.zeroes(@import("config.zig").Config);
    cfg.vocab_size = 64;
    cfg.hidden_size = 256;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 64;
    cfg.sliding_window = 128;
    cfg.log_n_floor = 256;
    cfg.log_alpha = 0.0;
    const S: c_int = 192;
    const q = try randomRow(&g, a, 7300, &.{ 1, S, 4, 64 }, 0.3, .bfloat16);
    const k = try randomRow(&g, a, 7301, &.{ 1, 2, S, 64 }, 0.3, .bfloat16);
    const v = try randomRow(&g, a, 7302, &.{ 1, 2, S, 64 }, 0.3, .bfloat16);
    const rel_states = try randomRow(&g, a, 7303, &.{ 1, S, 4, 16 }, 0.2, .bfloat16);
    const sinks = try randomRow(&g, a, 7304, &.{ 1, S, 4 }, 0.2, .bfloat16);
    const gate = try g.sigmoid(try randomRow(&g, a, 7305, &.{S * 4 * 64}, 1.0, .float32));
    var L: layers.Layer = undefined;
    L.rel_proj = try randomRow(&g, a, 7306, &.{ 16, 64 }, 0.1, .float32);
    L.rel_proj_t = .{};

    for ([_]bool{ true, false }) |is_global| {
        L.global = is_global;
        const raw = try kernels.relative(&g, &L, &cfg, q, rel_states, sinks, k, v, 0, true);
        const ref = try hostValues(a, &g, try kernels.gated(&g, raw, gate, S));
        defer a.free(ref);
        const got = try hostValues(a, &g, try kernels.relativeWith(&g, &L, &cfg, q, rel_states, sinks, k, v, 0, true, gate));
        defer a.free(got);
        try testing.expectEqualSlices(f32, ref, got);
    }
}

// Bar: fused `topKGroup` produces bit-identical picks, weights and expert tile descriptors to `topK` + `groupPicks`.
test "inkling2 kernels: topKGroup matches topK + groupPicks bit-for-bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = ops.Graph.init(a, s);
    defer g.deinit();

    const E: c_int = 128;
    const KSEL: c_int = 8;
    const rb = try randomRow(&g, a, 7500, &.{E}, 0.1, .float32);
    const rs = try g.fromSlice(f32, &.{1.25}, &.{1});
    for ([_]c_int{ 2, 3, 4, 8 }) |rows| {
        const scores = try randomRow(&g, a, 7510 + @as(u64, @intCast(rows)), &.{ rows, E }, 1.0, .float32);
        const ref_tk = try kernels.topK(&g, scores, rb, rs, KSEL);
        const ref_gr = try kernels.groupPicks(&g, try g.reshape(ref_tk[0], &.{rows * KSEL}), @intCast(E));
        const got = try kernels.topKGroup(&g, scores, rb, rs, KSEL, @intCast(E));
        for ([_][2]ops.Array{
            .{ ref_tk[0], got[0] }, .{ ref_tk[1], got[1] }, .{ ref_gr[0], got[2] }, .{ ref_gr[1], got[3] },
        }) |pair| {
            const rv = try hostValues(a, &g, pair[0]);
            defer a.free(rv);
            const gv = try hostValues(a, &g, pair[1]);
            defer a.free(gv);
            try testing.expectEqualSlices(f32, rv, gv);
        }
        const tcnt = try hostValues(a, &g, got[3]);
        defer a.free(tcnt);
        const rpick = try hostValues(a, &g, ref_gr[2]);
        defer a.free(rpick);
        const gpick = try hostValues(a, &g, got[4]);
        defer a.free(gpick);
        for (tcnt, 0..) |cnt_f, tile| {
            const cnt: usize = @intFromFloat(cnt_f);
            try testing.expectEqualSlices(f32, rpick[tile * 4 ..][0..cnt], gpick[tile * 4 ..][0..cnt]);
        }
    }
}

// Bar: group-64 8-bit quantized fold (`markovFoldQ8`) and fused `markovPickRnn` draft the op chain's tokens and
// probabilities (within 2e-2) across bf16 and f16, exercising both `quantizeFold` and the 15-launch fused chain.
test "inkling2 kernels: quantized markovFoldQ8 and fused markovPickRnn match the op chain" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const rank = 128;
    const hidden = 128;
    const vpad = 4096;
    const vocab = 4000;
    const steps = 7;
    for ([_]ops.Dtype{ .bfloat16, .float16 }) |dt| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const seed: u64 = if (dt == .bfloat16) 19100 else 19200;
        var d: draft.Drafter = .{
            .cfg = .{ .layers = 0, .heads = 1, .kv_heads = 1, .head_dim = 64, .intermediate = 1, .window = 2, .block = steps + 1, .rank = rank, .tap_count = 1 },
            .hidden = hidden,
            .s = s,
        };
        d.norm = try g.ones(&.{hidden}, dt);
        d.prev_rows = try randomRow(&g, a, seed, &.{ vpad, rank }, 0.5, dt);
        d.rnn = .{ .w = try randomRow(&g, a, seed + 1, &.{ 3 * rank, 2 * rank + 2 * hidden }, 0.05, dt) };
        d.rnn_bias = try randomRow(&g, a, seed + 2, &.{3 * rank}, 0.2, dt);
        d.fold = try randomRow(&g, a, seed + 3, &.{ rank, vpad }, 0.3, dt);
        d.fold_bias = try randomRow(&g, a, seed + 4, &.{ 1, 1, vpad }, 0.3, dt);
        const qf = try draft.quantizeFold(&g, d.fold, 64);
        d.fold_q = qf[0];
        d.fold_s = qf[1];
        d.fold_qb = qf[2];
        const rows = try randomRow(&g, a, seed + 5, &.{ 1, steps, hidden }, 1.0, dt);
        const newest = try randomRow(&g, a, seed + 6, &.{ 1, 1, hidden }, 1.0, dt);
        const base = try g.astype(try randomRow(&g, a, seed + 7, &.{ 1, steps, vpad }, 4.0, dt), .float32);
        const t1 = try g.fromSlice(i32, &.{1777}, &.{1});
        const want = try d.markov(&g, t1, rows, newest, base, vocab, 30, false, null);
        const got = try d.markov(&g, t1, rows, newest, base, vocab, 30, true, null);
        try testing.expectEqual(@as(f32, 0), try parity.maxAbsDiffF32(try g.astype(got.tokens, .float32), try g.astype(want.tokens, .float32), s));
        try testing.expect((try parity.maxAbsDiffF32(got.confidence, want.confidence, s)) <= 2e-2);
    }
}

// Bar: `downs_grp` fast `cnt == 1` path matches `layers.gather` across NVFP4, MXFP8 and affine 8-bit banks.
test "inkling2 kernels: downs_grp cnt == 1 fast path matches gather across packed formats" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const E: c_int = 16;
    const I: c_int = 256;
    const D: c_int = 256;
    const N: c_int = 4;
    const ModeCase = struct { mode: QuantMode, bits: u32, gs: u32, name: [*:0]const u8 };
    const cases = [_]ModeCase{
        .{ .mode = .nvfp4, .bits = 4, .gs = 16, .name = "nvfp4" },
        .{ .mode = .mxfp8, .bits = 8, .gs = 32, .name = "mxfp8" },
        .{ .mode = .affine, .bits = 8, .gs = 64, .name = "affine" },
    };
    for (cases, 0..) |mc, ci| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const raw = try randomRow(&g, a, 21000 + ci * 10, &.{ E, D, I }, 0.25, .bfloat16);
        var parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        try mlx.check(mlx.mlx_quantize(&parts, raw, mlx.mlx_optional_int.some(@intCast(mc.gs)), mlx.mlx_optional_int.some(@intCast(mc.bits)), mc.name, .{}, s));
        var bank: @import("../../sdk/bases/blocks/moe.zig").Bank = .{ .w = .{}, .qp = .{ .mode = mc.mode, .bits = mc.bits, .group_size = mc.gs } };
        var w0 = mlx.mlx_array_new();
        var s0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&w0, parts, 0));
        try mlx.check(mlx.mlx_vector_array_get(&s0, parts, 1));
        bank.w = try g.adopt(w0);
        bank.s = try g.adopt(s0);
        if (mc.mode == .affine) {
            var b0 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&b0, parts, 2));
            bank.b = try g.adopt(b0);
        }
        const act = try randomRow(&g, a, 21050 + ci * 10, &.{ N, 1, I }, 0.5, .bfloat16);
        // Distinct experts so every tile has cnt == 1.
        const sel = try g.fromSlice(u32, &.{ 1, 5, 9, 13 }, &.{N});
        const grp = try kernels.groupPicks(&g, sel, @intCast(E));
        const fmt = kernels.packedFormat(&bank, I).?;
        const wb = if (bank.b.ctx != null) bank.b else bank.s;
        const got = (try kernels.launchGrid(&g, &kernels.downs_grp_kernel, &.{ act, bank.w, bank.s, wb, grp[0], grp[1], grp[2] }, 1, .{.{ .shape = &.{ N, 1, D }, .dtype = .bfloat16 }}, .{ 32 * N, @divExact(D, 4), 1 }, .{ 32, 4, 1 }, &.{
            .{ "T", .{ .dtype = .bfloat16 } },         .{ "I", .{ .int = I } },                          .{ "D", .{ .int = D } },
            .{ "GS", .{ .int = @intCast(mc.gs) } },    .{ "FMT", .{ .int = fmt } },                      .{ "BITS", .{ .int = @intCast(mc.bits) } },
            .{ "S", .{ .dtype = ops.dtype(bank.s) } }, .{ "BM", .{ .int = kernels.groupTileWidth(N) } },
        }))[0];
        const want = try layers.gather(&g, act, &bank, sel, false);
        try testing.expect((try parity.maxAbsDiffF32(got, want, s)) <= 5e-2);
    }
}

// Bar: fused sampled Markov chain (`LAW=1`: `markovFoldQ8Law` + `markovPickRnnLaw` + parallel `exp(scaled - lse)`)
// returns valid probability rows summing to 1, assigns positive mass to each drawn token, reports `max(probs)` as
// confidence, and collapses to exact argmax at cold temperature (1e-4).
test "inkling2 kernels: fused LAW=1 sampled Markov chain returns valid law and cold argmax" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const saved = fx_stream;
    fx_stream = s;
    defer fx_stream = saved;
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const rank = 128;
    const hidden = 128;
    const vpad = 4096;
    const vocab = 4000;
    const steps = 5;
    const dt: ops.Dtype = .bfloat16;
    var d: draft.Drafter = .{
        .cfg = .{ .layers = 0, .heads = 1, .kv_heads = 1, .head_dim = 64, .intermediate = 1, .window = 2, .block = steps + 1, .rank = rank, .tap_count = 1 },
        .hidden = hidden,
        .s = s,
    };
    d.norm = try g.ones(&.{hidden}, dt);
    d.prev_rows = try randomRow(&g, a, 22100, &.{ vpad, rank }, 0.5, dt);
    d.rnn = .{ .w = try randomRow(&g, a, 22101, &.{ 3 * rank, 2 * rank + 2 * hidden }, 0.05, dt) };
    d.rnn_bias = try randomRow(&g, a, 22102, &.{3 * rank}, 0.2, dt);
    d.fold = try randomRow(&g, a, 22103, &.{ rank, vpad }, 0.3, dt);
    d.fold_bias = try randomRow(&g, a, 22104, &.{ 1, 1, vpad }, 0.3, dt);
    const qf = try draft.quantizeFold(&g, d.fold, 64);
    d.fold_q = qf[0];
    d.fold_s = qf[1];
    d.fold_qb = qf[2];
    const rows = try randomRow(&g, a, 22105, &.{ 1, steps, hidden }, 1.0, dt);
    const newest = try randomRow(&g, a, 22106, &.{ 1, 1, hidden }, 1.0, dt);
    const base = try g.astype(try randomRow(&g, a, 22107, &.{ 1, steps, vpad }, 4.0, dt), .float32);
    const t1 = try g.fromSlice(i32, &.{1777}, &.{1});

    const argmax = try d.markov(&g, t1, rows, newest, base, vocab, 30, true, null);
    const cold = try d.markov(&g, t1, rows, newest, base, vocab, 30, true, .{ .temperature = 1e-4, .seed = 7 });
    try testing.expectEqual(@as(f32, 0), try parity.maxAbsDiffF32(try g.astype(cold.tokens, .float32), try g.astype(argmax.tokens, .float32), s));

    const warm = try d.markov(&g, t1, rows, newest, base, vocab, 30, true, .{ .temperature = 1.0, .seed = 11 });
    const law = try hostValues(a, &g, warm.probs);
    defer a.free(law);
    const sure = try hostValues(a, &g, warm.confidence);
    defer a.free(sure);
    const ids = try hostValues(a, &g, try g.astype(warm.tokens, .float32));
    defer a.free(ids);
    try testing.expectEqual(@as(usize, steps * vocab), law.len);
    for (0..steps) |i| {
        const row = law[i * vocab ..][0..vocab];
        var sum: f32 = 0;
        for (row) |x| sum += x;
        try testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-4);
        try testing.expect(row[@intFromFloat(ids[i])] > 0);
        try testing.expectApproxEqAbs(std.mem.max(f32, row), sure[i], 1e-4);
    }
}
